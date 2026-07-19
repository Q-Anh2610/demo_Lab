# Sinh caption tiếng Việt & Đánh giá chất lượng cho bộ dữ liệu Canifa

## 1. Vấn đề cần giải quyết

Caption gốc thu thập cùng ảnh sản phẩm trên Canifa (qua crawl) chứa các thông tin **không thể suy ra
chỉ từ ảnh** — ví dụ thành phần chất liệu vải (`60% cotton, 40% polyester`), tên bộ sưu tập, mùa ra mắt.
Nếu dùng trực tiếp caption này để huấn luyện/đánh giá mô hình captioning, mô hình sẽ bị buộc phải
"bịa" ra thông tin nó không thể nhìn thấy trong ảnh.

**Mục tiêu:** sinh một bộ caption tiếng Việt **mới**, tổng hợp từ 3 ảnh chụp nhiều góc của cùng 1 sản
phẩm, chỉ mô tả nội dung quan sát được (loại trang phục, kiểu dáng, màu sắc, chi tiết thiết kế).

## 2. Dữ liệu đầu vào
- Tổng **1.883 sản phẩm × 3 ảnh** (mỗi sản phẩm 3 góc chụp), phân bố theo
  `gender_category` (nu/nam/be_gai/be_trai) và `category1` (ao-phong, quan-shorts, bo-quan-ao, quan,
  vay, ...).
- Metadata đầu vào dạng CSV **wide** — mỗi dòng 1 sản phẩm, kèm sẵn 3 cột đường dẫn ảnh
  (`image_1_local_path`, `image_2_local_path`, `image_3_local_path`) và caption gốc.

## 3. Sinh caption bằng Qwen2-VL-7B-Instruct — `gen_caption_qwen.ipynb`

| Thành phần | Giá trị |
|---|---|
| Model | `Qwen/Qwen2-VL-7B-Instruct` |
| Lượng tử hoá | 4-bit NF4 qua `bitsandbytes`, tính toán ở fp16 (GPU T4 — kiến trúc Turing không có bf16 tensor core) |
| VRAM | ~5–6GB cho trọng số → vừa thoải mái trong 15GB của 1 GPU T4 trên Kaggle |
| Input/Output | 3 ảnh cùng 1 sản phẩm → **1 lần gọi model** → 1 caption (không sinh riêng từng ảnh rồi ghép) |
| Prompt | Yêu cầu model đóng vai copywriter thời trang, viết caption tiếng Việt đúng 2–3 câu, dựa trên 3 ảnh |

**Quy trình kỹ thuật đáng chú ý:**
- Sau khi sinh xong, kết quả được merge với metadata gốc theo `product_slug`, tạo ra 1 CSV có **cả 2
  cột**: `caption` (gốc, từ crawl — có thể chứa chất liệu/BST không thấy từ ảnh) và `caption_synth`
  (mới, sinh từ ảnh) để tiện đối chiếu ở bước lọc tiếp theo.
- Kết quả cuối được đẩy lên HuggingFace Hub (`qa994/canifa-captions`) để tách khỏi phiên Kaggle, dùng
  lại được ở bước review offline trên máy Windows cá nhân.

## 4. Quy trình kiểm định chất lượng kết quả caption — 4 bước đã triển khai

| Bước | Mô tả | Thực hiện ở đâu |
|---|---|---|
| 1. Rule-based | Quét từ khoá cấm (chất liệu vải, tên BST...) | Tự động, áp dụng cho toàn bộ 1.883 mẫu |
| 2. Embedding similarity | So khớp ảnh–văn bản kiểu CLIP, phát hiện caption lệch nội dung ảnh | Tự động, áp dụng cho toàn bộ 1.883 mẫu |
| 3. Manual spot-check | Review thủ công qua giao diện Gradio | `canifa_caption_review_app.py` |
| 4. Decision logging | Ghi log quyết định, loại lỗi, caption đã sửa | `review_decisions.csv` |

100 mẫu ở bước 3 được lấy mẫu **stratified** theo `gender_category × category1` (đảm bảo các nhóm nhỏ
như `be_trai` vẫn có đại diện, thay vì random thuần có thể bỏ sót) và dùng cho 3 mục đích:

1. Làm **gold set** để tính BLEU/ROUGE-L/CIDEr về sau (cần caption "chuẩn" do người xác nhận).
2. Hiệu chỉnh ngưỡng (threshold) cho bước 1–2 (rule-based & embedding similarity).
3. Phát hiện lỗi hệ thống mang tính lặp lại để tinh chỉnh prompt sinh caption

### 4.2. Giao diện review — `canifa_caption_review_app.py`

Gradio app chạy **local trên máy Windows** (nơi có sẵn thư mục ảnh), không cần deploy:

- Hiển thị song song: 3 ảnh sản phẩm, caption gốc (crawl), caption sinh ra (Qwen2-VL), và 1 ô để sửa
  trực tiếp caption nếu cần.
- 3 lựa chọn quyết định: **"Đạt - dùng nguyên caption sinh"**, **"Đạt sau khi sửa"**,
  **"Không đạt - loại bỏ"**.
- Checklist 8 loại lỗi để gắn nhãn khi caption có vấn đề: sai loại trang phục, sai màu sắc, sai giới
  tính/đối tượng mặc, thiếu chi tiết quan trọng, bịa thông tin không thấy trong ảnh (hallucination),
  lỗi ngữ pháp/câu văn không tự nhiên, quá chung chung không có điểm nhấn, khác.
- Mỗi lượt review được ghi ngay vào `review_decisions.csv` (có timestamp) — cơ chế **resume**: mở lại
  app sẽ tự đọc file này để biết mẫu nào đã review, không hiện lại.
- Điều hướng bằng cách bấm trực tiếp vào dòng trong bảng mục lục (không chỉ next/prev tuần tự).