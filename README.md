# 🧠 LSTM From Scratch using TensorFlow

Xây dựng mô hình LSTM hoàn toàn từ đầu (không dùng tf.keras.layers.LSTM), huấn luyện và đánh giá trên bộ dữ liệu IMDB Sentiment Classification.

## 📘 Giới thiệu

Repo này minh họa cách tự xây dựng một mô hình LSTM từ đầu bằng cách cài đặt toàn bộ các phép tính toán của các cổng LSTM (Input, Forget, Output, Candidate) trong lớp tf.keras.layers.Layer.

Mục tiêu là giúp hiểu cơ chế nội tại của LSTM — cách mà nó lưu trữ, quên và cập nhật thông tin theo chuỗi — thay vì chỉ sử dụng lớp LSTM có sẵn trong TensorFlow.

Mô hình được huấn luyện thử nghiệm trên bộ IMDB Reviews Dataset, một tập dữ liệu phổ biến trong bài toán phân loại cảm xúc (sentiment analysis).

## ⚙️ Cấu trúc dự án
```
📂 lstm_from_scratch/
│
├── 📜 LSTM_From_Scratch.py         # Cài đặt lớp LSTM thủ công
├── 📜 RNN_with_LSTM_From_Scratch.py # Mô hình RNN kết hợp lớp LSTM tự build
├── 📜 train_imdb.py                 # File huấn luyện và đánh giá mô hình
├── 📜 requirements.txt              # Danh sách thư viện cần thiết
└── 📘 README.md                     # File giới thiệu (bạn đang đọc đây)
```

## 🧩 Thành phần chính
### 1. LSTM_From_Scratch

Tự cài đặt một cell LSTM với 4 cổng:

Input Gate – Quyết định bao nhiêu thông tin mới được thêm vào cell state.

Forget Gate – Học xem nên quên bao nhiêu phần thông tin cũ.

Output Gate – Quyết định bao nhiêu thông tin từ cell state được đưa ra làm hidden state.

Candidate Cell (n_c_t) – Sinh thông tin mới để kết hợp vào cell state.

Toàn bộ phép nhân ma trận, cộng, sigmoid, tanh đều được tính thủ công bằng TensorFlow.

### 2. LSTM_From_Scratch_Model

Kết hợp lớp Embedding, lớp LSTM tự cài đặt, và một mạng Fully Connected để phân loại.

Mỗi bước thời gian (timestep) sẽ truyền hidden state và cell state qua LSTM_From_Scratch.

Kết quả cuối cùng (hidden state cuối) được đưa qua Dense layers để dự đoán cảm xúc.

## 📊 Dữ liệu huấn luyện

Dataset: IMDB Reviews (có sẵn trong tf.keras.datasets.imdb)

Nhiệm vụ: Dự đoán cảm xúc tích cực (1) hoặc tiêu cực (0).

Tiền xử lý:

Token hóa và ánh xạ từ sang số.

Padding các chuỗi về cùng độ dài (ví dụ: maxlen=200).

## 🚀 Cách chạy thử
1️⃣ Cài đặt môi trường
```bash
git clone https://github.com/PhuIT2503/LSTM-From-Scratch
cd lstm-from-scratch
pip install -r requirements.txt
```
2️⃣ Huấn luyện mô hình
Chạy file LSTM_From_Scratch.ipynb

3️⃣ Kết quả dự kiến

Sau vài epoch (tuỳ cấu hình), mô hình 86% accuracy trên tập test, đủ để chứng minh rằng mô hình LSTM tự cài đặt hoạt động đúng.

## 🧠 Ý nghĩa học thuật

Repo này được thiết kế với mục đích giáo dục và trực quan hóa cơ chế LSTM, không tập trung vào tối ưu hiệu năng.
Bạn có thể:

Thay đổi kích thước embedding hoặc units.

Quan sát các giá trị cổng (gate activations) để hiểu cách mô hình ghi nhớ thông tin.

So sánh với tf.keras.layers.LSTM để thấy sự khác biệt về tốc độ và kết quả.

## 📚 Tham khảo

Understanding LSTM Networks — Colah’s Blog

TensorFlow RNN Guide

IMDB Dataset on TensorFlow

## ✨ Tác giả

**Phan Quyết Tâm Phú**  
AI Engineer (NLP & Machine Learning)  
📧 Liên hệ: tamphu.workhard@gmail.com
🌐 GitHub: https://github.com/PhuIT2503

“Học cách mô hình hoạt động bên trong giúp ta làm chủ công nghệ, không chỉ sử dụng nó.”