"""Gradio app de review va sua caption sinh tu Qwen cho du an Canifa Image Captioning.

Chay LOCAL tren may Windows cua ban (noi co san thu muc anh D:\\canifa_dataset\\images):
    pip install gradio pandas
    python canifa_caption_review_app.py

App se mo tai http://127.0.0.1:7860

Vai tro trong quy trinh 5 buoc loc caption:
    1. Rule-based filter       -> ap dung tu dong cho ca 1,883 mau (KHONG lam trong app nay)
    2. Embedding similarity    -> ap dung tu dong cho ca 1,883 mau (KHONG lam trong app nay)
    3. Manual spot-check       -> APP NAY: nguoi review 1 mau nho (mac dinh 100, stratified
                                  theo gender_category x category1) de: (a) tao gold set tinh
                                  BLEU/CIDEr/ROUGE-L, (b) hieu chinh threshold cho buoc 1-2,
                                  (c) rut ra pattern loi de tinh chinh prompt / chon mau can
                                  regenerate, KHONG phai de sua tay het 1,883 mau.
    4. Decision logging        -> APP NAY ghi lai vao review_decisions.csv (co timestamp,
                                  quyet dinh, loai loi, ghi chu) de truy vet lai duoc.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("caption_review_app")


REQUIRED_COLUMNS = [
    "product_slug", "gender_category", "category1", "caption", "caption_synth",
    "image_1_local_path", "image_2_local_path", "image_3_local_path",
]

DECISION_OPTIONS = [
    "Dat - dung nguyen caption sinh",
    "Dat sau khi sua",
    "Khong dat - loai bo",
]

ERROR_TAG_OPTIONS = [
    "Sai loai trang phuc",
    "Sai mau sac",
    "Sai gioi tinh / doi tuong mac",
    "Thieu chi tiet quan trong",
    "Bia thong tin khong thay trong anh (hallucination)",
    "Loi ngu phap / cau van khong tu nhien",
    "Qua chung chung, khong co diem nhan",
    "Khac",
]

REVIEW_COLUMNS = [
    "product_slug", "reviewed_at", "decision", "error_tags",
    "generated_caption", "corrected_caption", "notes",
]


@dataclass(frozen=True)
class AppConfig:
    """Cau hinh app, khong hardcode duong dan - truyen qua CLI argument."""

    metadata_csv: Path
    reviews_csv: Path
    sample_size: int = 100
    seed: int = 42
    stratify_columns: tuple[str, ...] = ("gender_category", "category1")


def set_seed(seed: int) -> None:
    random.seed(seed)


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    """Doc CSV metadata (ban da merge caption_synth vao roi tai ve tu Kaggle/HF).

    Raises:
        FileNotFoundError: neu khong tim thay file CSV.
        KeyError: neu CSV thieu cot bat buoc.
    """
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Khong tim thay metadata CSV: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise KeyError(f"CSV thieu cac cot bat buoc: {missing_cols}")

    n_no_caption = df["caption_synth"].isna().sum()
    if n_no_caption:
        logger.warning(
            "%d san pham chua co caption_synth (fail o buoc sinh caption) - se KHONG dua "
            "vao mau review.", n_no_caption
        )
    return df[df["caption_synth"].notna()].reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int,
                       stratify_columns: tuple[str, ...]) -> pd.DataFrame:
    """Lay mau stratified theo ty le phan bo cua cac cot trong stratify_columns.

    Dam bao cac nhom nho (vi du gender_category='be_trai') van co dai dien trong mau review,
    thay vi random thuan co the bo sot.
    """
    missing = [c for c in stratify_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Khong tim thay cot stratify: {missing}")

    if sample_size >= len(df):
        logger.info("sample_size (%d) >= tong so san pham hop le (%d), lay het.",
                     sample_size, len(df))
        return df.copy()

    group_cols = list(stratify_columns)
    frac = sample_size / len(df)

    # QUAN TRONG: KHONG dung df.groupby(...).apply(...) o day. Cac ban pandas gan day
    # (>=2.2, tuy phien ban co the da doi default cua tham so `include_groups`) co the
    # LOAI BO cac cot dung de group ra khoi DataFrame duoc truyen vao ham apply, khien
    # cot stratify (vi du gender_category) bien mat khoi ket qua sau cung ma khong bao
    # loi ngay tai day - loi chi lo ra sau, luc truy cap row['gender_category'].
    # Duyet thu cong qua tung group (iterate groupby object) de dam bao luon giu du cot.
    group_frames = []
    for _, group_df in df.groupby(group_cols, group_keys=False):
        n = max(1, round(len(group_df) * frac))
        n = min(n, len(group_df))
        group_frames.append(group_df.sample(n=n, random_state=seed))
    sampled = pd.concat(group_frames, ignore_index=False)

    # Cat/bu cho dung sample_size neu lam tron gay lech (thuong lech 1-3 mau)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed)
    elif len(sampled) < sample_size:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(
            n=min(sample_size - len(sampled), len(remaining)), random_state=seed
        )
        sampled = pd.concat([sampled, extra])

    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)  # xao thu tu

    # Kiem tra lai chac chan cot stratify van con - fail nhanh, ro rang thay vi loi ngam
    # sau nay o render() giong lan truoc.
    missing_after = [c for c in stratify_columns if c not in sampled.columns]
    if missing_after:
        raise RuntimeError(
            f"Loi noi bo: cac cot stratify {missing_after} bi mat sau khi sampling. "
            f"Kiem tra lai phien ban pandas ({pd.__version__}) va logic groupby o day."
        )

    logger.info("Da lay stratified sample: %d san pham (tu %d hop le).", len(sampled), len(df))
    return sampled


def load_reviewed_slugs(reviews_csv: Path) -> set[str]:
    """Doc danh sach product_slug da review de resume (khong review lai)."""
    if not reviews_csv.exists():
        return set()
    try:
        df = pd.read_csv(reviews_csv)
        return set(df["product_slug"].tolist())
    except (pd.errors.EmptyDataError, KeyError) as e:
        logger.warning("File review rong hoac thieu cot, coi nhu chua review gi: %s", e)
        return set()


def append_review(reviews_csv: Path, record: dict) -> None:
    """Ghi 1 dong quyet dinh review vao CSV (append-only), tao header neu file chua ton tai."""
    file_exists = reviews_csv.exists()
    reviews_csv.parent.mkdir(parents=True, exist_ok=True)
    with reviews_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def resolve_image_path(local_path: str) -> Optional[str]:
    """Tra ve duong dan anh neu ton tai tren dia, None neu khong tim thay (de UI hien placeholder)."""
    try:
        p = Path(str(local_path))
    except (TypeError, ValueError) as e:
        logger.error("Duong dan anh khong hop le: %s (%s)", local_path, e)
        return None
    return str(p) if p.exists() else None


def build_index_df(state_df: pd.DataFrame, reviews_csv: Path, current_idx: int) -> pd.DataFrame:
    """Tao bang muc luc (STT, product_slug, trang thai) de nguoi dung bam chon nhay toi.

    Trang thai lay tu review_decisions.csv (doc lai moi lan goi de luon cap nhat sau khi luu).
    """
    if reviews_csv.exists():
        try:
            reviews_df = pd.read_csv(reviews_csv).set_index("product_slug")["decision"]
        except (pd.errors.EmptyDataError, KeyError):
            reviews_df = pd.Series(dtype=str)
    else:
        reviews_df = pd.Series(dtype=str)

    rows = []
    for i, (_, row) in enumerate(state_df.iterrows()):
        slug = row["product_slug"]
        status = reviews_df.get(slug, "Chua review")
        marker = ">> " if i == current_idx else "   "
        rows.append({"STT": marker + str(i + 1), "product_slug": slug, "Trang thai": status})
    return pd.DataFrame(rows)


def build_app(cfg: AppConfig) -> gr.Blocks:
    full_df = load_metadata(cfg.metadata_csv)
    sample_df = stratified_sample(full_df, cfg.sample_size, cfg.seed, cfg.stratify_columns)

    reviewed = load_reviewed_slugs(cfg.reviews_csv)
    pending_mask = ~sample_df["product_slug"].isin(reviewed)
    logger.info("Da review truoc do: %d | Con lai trong mau: %d",
                len(reviewed), pending_mask.sum())

    state_df = sample_df  # giu nguyen ca sample de co the xem lai muc da review

    with gr.Blocks(title="Canifa Caption Review") as demo:
        gr.Markdown("# Review caption sinh tu anh san pham (Canifa)")

        with gr.Row():
            # --- Cot trai: muc luc, bam vao dong de nhay thang toi san pham do ---
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Muc luc (bam vao 1 dong de nhay toi)")
                index_table = gr.Dataframe(
                    headers=["STT", "product_slug", "Trang thai"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    row_count=(cfg.sample_size, "fixed"),
                )

            # --- Cot phai: khu vuc review nhu cu ---
            with gr.Column(scale=3):
                progress_md = gr.Markdown()

                with gr.Row():
                    img1 = gr.Image(label="Anh 1", type="filepath", height=280)
                    img2 = gr.Image(label="Anh 2", type="filepath", height=280)
                    img3 = gr.Image(label="Anh 3", type="filepath", height=280)

                product_info_md = gr.Markdown()

                original_caption_box = gr.Textbox(
                    label="Caption GOC (tu crawl - chi de doi chieu, CO THE chua thong tin khong "
                          "thay tu anh nhu chat lieu vai)",
                    interactive=False, lines=3,
                )
                generated_caption_box = gr.Textbox(
                    label="Caption SINH RA (Qwen2-VL)", interactive=False, lines=3,
                )
                corrected_caption_box = gr.Textbox(
                    label="Caption SAU KHI SUA (chinh sua truc tiep o day, mac dinh copy tu "
                          "caption sinh ra)",
                    lines=3,
                )

                decision_radio = gr.Radio(choices=DECISION_OPTIONS, label="Quyet dinh")
                error_tags_group = gr.CheckboxGroup(choices=ERROR_TAG_OPTIONS,
                                                     label="Loai loi (neu co)")
                notes_box = gr.Textbox(label="Ghi chu them (tuy chon)", lines=2)

                with gr.Row():
                    prev_btn = gr.Button("<< Quay lai")
                    skip_btn = gr.Button("Bo qua (khong luu)")
                    save_next_btn = gr.Button("Luu & tiep theo >>", variant="primary")

                status_md = gr.Markdown()

        current_idx = gr.State(value=0)

        def find_next_unreviewed(start_idx: int, direction: int = 1) -> int:
            """Tim index tiep theo (hoac truoc do) CHUA duoc review, tu start_idx."""
            n = len(state_df)
            idx = start_idx
            for _ in range(n):
                slug = state_df.iloc[idx]["product_slug"]
                if slug not in load_reviewed_slugs(cfg.reviews_csv):
                    return idx
                idx = (idx + direction) % n
            return start_idx  # tat ca da review het

        def render(idx: int):
            row = state_df.iloc[idx]
            reviewed_now = load_reviewed_slugs(cfg.reviews_csv)
            n_reviewed = len(reviewed_now & set(state_df["product_slug"]))

            progress_text = (
                f"**Tien do: {n_reviewed}/{len(state_df)} da review** "
                f"(dang xem #{idx + 1}/{len(state_df)})"
            )
            info_text = (
                f"**product_slug:** `{row['product_slug']}`  |  "
                f"**gender_category:** {row['gender_category']}  |  "
                f"**category1:** {row['category1']}"
            )

            paths = [resolve_image_path(row[c]) for c in
                     ("image_1_local_path", "image_2_local_path", "image_3_local_path")]
            for p, col in zip(paths, ("image_1_local_path", "image_2_local_path", "image_3_local_path")):
                if p is None:
                    logger.warning("Khong tim thay anh tren dia cho %s (%s)",
                                   row["product_slug"], col)

            gen_caption = row["caption_synth"]
            index_df = build_index_df(state_df, cfg.reviews_csv, idx)

            return (
                progress_text, paths[0], paths[1], paths[2], info_text,
                row["caption"], gen_caption, gen_caption,  # pre-fill corrected = generated
                None, [], "",  # reset decision / tags / notes
                idx, index_df,
            )

        def go_to(idx: int):
            return render(idx)

        def on_prev(idx: int):
            new_idx = (idx - 1) % len(state_df)
            return render(new_idx)

        def on_skip(idx: int):
            new_idx = find_next_unreviewed((idx + 1) % len(state_df))
            return render(new_idx)

        def on_pick_row(evt: gr.SelectData):
            """Bam vao 1 dong trong muc luc -> nhay thang toi san pham do."""
            picked_idx = evt.index[0]
            return render(picked_idx)

        def on_save_next(idx: int, decision: Optional[str], error_tags: list,
                          corrected_caption: str, notes: str):
            row = state_df.iloc[idx]

            if not decision:
                status = "**Chua chon Quyet dinh - vui long chon truoc khi luu.**"
                current_render = render(idx)
                return (*current_render, status)

            record = {
                "product_slug": row["product_slug"],
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "decision": decision,
                "error_tags": "|".join(error_tags) if error_tags else "",
                "generated_caption": row["caption_synth"],
                "corrected_caption": corrected_caption,
                "notes": notes,
            }
            try:
                append_review(cfg.reviews_csv, record)
            except OSError as e:
                logger.error("Khong ghi duoc file review: %s", e)
                status = f"**Loi khi luu: {e}**"
                current_render = render(idx)
                return (*current_render, status)

            logger.info("Da luu review cho %s: %s", row["product_slug"], decision)
            new_idx = find_next_unreviewed((idx + 1) % len(state_df))
            new_render = render(new_idx)
            status = f"**Da luu review cho `{row['product_slug']}`.**"
            return (*new_render, status)

        outputs_common = [
            progress_md, img1, img2, img3, product_info_md,
            original_caption_box, generated_caption_box, corrected_caption_box,
            decision_radio, error_tags_group, notes_box, current_idx, index_table,
        ]

        demo.load(fn=lambda: go_to(find_next_unreviewed(0)), outputs=outputs_common)
        prev_btn.click(fn=on_prev, inputs=[current_idx], outputs=outputs_common)
        skip_btn.click(fn=on_skip, inputs=[current_idx], outputs=outputs_common)
        index_table.select(fn=on_pick_row, outputs=outputs_common)
        save_next_btn.click(
            fn=on_save_next,
            inputs=[current_idx, decision_radio, error_tags_group,
                    corrected_caption_box, notes_box],
            outputs=outputs_common + [status_md],
        )

    return demo


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Canifa caption review app")
    parser.add_argument("--metadata-csv", type=Path, required=True,
                         help="Duong dan CSV da merge caption_synth (tai ve tu Kaggle/HF)")
    parser.add_argument("--reviews-csv", type=Path, default=Path("./review_decisions.csv"),
                         help="Noi luu ket qua review (mac dinh: ./review_decisions.csv)")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return AppConfig(
        metadata_csv=args.metadata_csv,
        reviews_csv=args.reviews_csv,
        sample_size=args.sample_size,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    demo = build_app(cfg)
    demo.launch(allowed_paths=[str(cfg.metadata_csv.parent)])

if __name__ == "__main__":
    main()
