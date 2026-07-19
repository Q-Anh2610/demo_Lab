# Tuần 6 — So sánh 4 phương pháp Knowledge Distillation & Ablation số tầng Decoder

Thư mục này ghi lại toàn bộ quá trình thực nghiệm của tuần 6, gồm 2 mục tiêu chính:

1. So sánh **4 kỹ thuật Knowledge Distillation (KD)** cho bài toán Image Captioning thời trang:
   **Word-KD (response-based)** kết hợp lần lượt với **FitNets, Attention Transfer (AT), MGD, SPKD** (feature-based).
2. Sau khi chọn ra phương pháp tốt nhất (**Word-KD + Attention Transfer**), thực hiện **ablation study**
   để đo ảnh hưởng riêng của việc **prune số tầng decoder** (4 / 6 / 12 tầng) tới chất lượng caption.

Tất cả notebook được chạy trên **Kaggle**. Dataset dùng chung được build một lần và đẩy lên HuggingFace Hub để đảm bảo mọi notebook
train/eval trên đúng cùng 1 tập dữ liệu.

---

## 1. Dữ liệu — `dataset10k.ipynb`

- Nguồn: [`Marqo/fashion200k`](https://huggingface.co/datasets/Marqo/fashion200k) trên HuggingFace Hub.
- Lấy cố định **10.000 ảnh**, chia theo tỉ lệ **7 / 1 / 2** (train / val / test).
- Gom nhóm chống rò rỉ dữ liệu bằng **Union-Find** đảm bảo cùng 1 sản phẩm không xuất hiện ở cả train và test.
- Chia theo đơn vị nhóm sản phẩm.
- Output: mỗi split gồm `images/<item_ID>.jpg` + `metadata.json`, sau đó đẩy lên HuggingFace: https://huggingface.co/datasets/qa994/fashion200k_10k —
  đây là **nguồn dữ liệu duy nhất** mà 6 notebook còn lại trong tuần này sử dụng.

---

## 2. So sánh 4 phương pháp KD — `fitnets_at.ipynb` + `mgd_spkd.ipynb`

Cả hai notebook dùng chung một pipeline:

- **Teacher (dùng chung, huấn luyện 1 lần):** BLIP-large, vision encoder đóng băng, fine-tune decoder.
- **Student (4 biến thể, kiến trúc giống hệt nhau):** BLIP-base, decoder đã prune còn 4 tầng
  (giữ so le layer `0, 3, 6, 9`).
- Cả 4 biến thể đều bật **Word-KD (response-based)** làm nền, khác nhau ở phần **feature-based**
  đi kèm:
  - `fitnets_at.ipynb`: **FitNets** và **Attention Transfer**
  - `mgd_spkd.ipynb`: **Masked Generative Distillation** và **Similarity-Preserving KD**
- Metrics đo: BLEU, ROUGE-1/2/L, METEOR, CIDEr, cùng bộ đo runtime (kích thước model, VRAM, latency,
  throughput) — dùng chung 1 hàm đo cho Teacher và cả 4 Student để đảm bảo so sánh công bằng.

> Phần **FitNets** trong `fitnets_at.ipynb` bị lỗi
> cân bằng loss, khiến phần đánh giá định tính lẫn định lượng của riêng FitNets ở notebook này
> **chưa đáng tin cậy**. Bug này được xác định và fix ở bước tiếp theo.

---

## 3. Fix bug FitNets & Tổng hợp so sánh — `evaluate.ipynb`

Notebook gồm 2 phần:

- **Phần A:** Sửa `FitNetsMultiLayer` — thêm chuẩn hoá L2 (`F.normalize`) lên đặc trưng Student
  (sau regressor) và đặc trưng Teacher **trước khi** tính MSE. Train lại **riêng FitNets**
  đủ 6 epoch.
- **Phần B:** Tổng hợp so sánh định lượng (BLEU/CIDEr/ROUGE/METEOR) và định tính (đọc caption mẫu)
  của cả **4 phương pháp** sau khi đã sửa FitNets.

**Kết luận:** phương pháp **Attention Transfer + Word-KD** cho kết quả tốt nhất trong 4
phương pháp, được chọn làm hướng đi tiếp theo cho ablation số tầng decoder ở phần 4.

---

## 4. Ablation số tầng Decoder — `4layers.ipynb`, `6layers.ipynb`, `12layers.ipynb`

Cả 3 notebook giữ nguyên pipeline/hyperparameter, chỉ thay đổi số tầng decoder giữ lại:

| Notebook | Số tầng decoder | Layer giữ lại |
|---|---|---|
| `4layers.ipynb` | 4 / 12 | `0, 3, 6, 9` |
| `6layers.ipynb` | 6 / 12 | so le |
| `2layers.ipynb` | 12 / 12 (không prune) | tất cả |

---

## 5. Kết quả

- `results/4methods_comparison_metrics.png` - so sánh metrics 4 phương pháp KD
- `results/4methods_comparison_latency.png` - so sánh kích thước và latency 4 phương pháp 
- `results/metrics/decoder_ablation.png` - so sánh metrics 4/6/12 tầng
- `results/fig_efficiency_depth_ablation.png` - so sánh hiệu năng khi prune 4/6/12 tầng


