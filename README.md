## 1. Giới thiệu đề tài
Dự án sử dụng Machine Learning để dự đoán giá bán của xe hơi dựa trên các đặc điểm kỹ thuật. Đây là bài toán **Regression** (Hồi quy).
## 2. Dataset
- https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge Dữ liệu từ file `data.csv`.
- Mô tả: Gồm 19,237 dòng và 18 cột (Hãng xe, Năm sản xuất, Dung tích động cơ, Số km đã đi, v.v.).

## 3. Pipeline
- Tiền xử lý: Xử lý dữ liệu trống, chuyển đổi cột `Mileage` từ chuỗi sang số, xử lý ngoại lệ (outliers).
- Feature Engineering: Sử dụng One-hot encoding cho các biến phân loại.
- Huấn luyện: Chia tập dữ liệu (80% Train - 20% Test).
- Mô hình: `RandomForestRegressor`.
- Triển khai: Web App bằng Streamlit.
- Đánh giá: Tính toán R² score, MAE và RMSE.

## 4. Kết quả
- Mô hình đạt chỉ số R² cao trên tập kiểm tra, dự báo sát với giá thực tế.

## 5. Hướng dẫn chạy dự án
1. Cài đặt thư viện: `pip install -r requirements.txt`
2. Chạy Train/Demo: Mở file trong thư mục `AI-ML` và chạy.
3. Chạy Web App: `streamlit run app/app.py`

## 6. Tác giả
- Họ tên: Lê Nguyễn Thái Dương
- MSSV: 12423008
- Lớp: 12423TN