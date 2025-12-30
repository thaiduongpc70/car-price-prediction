import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Dự đoán giá xe hơi")
st.write("Ứng dụng dự đoán giá xe dựa trên mô hình Random Forest từ AI-ML.ipynb")

# Load model và columns
# Lưu ý: Đảm bảo bạn đã export file columns.pkl từ notebook bằng lệnh: 
# joblib.dump(X_train.columns.tolist(), "columns.pkl")
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.sidebar.header("Thông tin xe")

# 1. Manufacturer (Hãng xe)
manufacturer = st.sidebar.selectbox(
    "Hãng xe",
    ["TOYOTA", "HYUNDAI", "HONDA", "LEXUS", "FORD", "CHEVROLET", "MERCEDES-BENZ", "BMW", "KIA"]
)

# 2. Fuel type (Loại nhiên liệu)
fuel_type = st.sidebar.selectbox(
    "Loại nhiên liệu",
    ["Petrol", "Diesel", "Hybrid", "LPG", "CNG", "Hydrogen"]
)

# 3. Gear box type (Hộp số)
gear_box = st.sidebar.selectbox(
    "Hộp số",
    ["Automatic", "Tiptronic", "Manual", "Variator"]
)

# 4. Mileage (Số km)
mileage_val = st.sidebar.number_input(
    "Số km đã chạy",
    min_value=0,
    max_value=1000000,
    value=50000,
    step=1000
)

# 5. Engine volume (Dung tích máy)
engine = st.sidebar.number_input(
    "Dung tích động cơ (L)",
    min_value=0.1,
    max_value=20.0,
    value=2.0,
    step=0.1
)

# 6. Prod. year (Năm sản xuất) - Trong Notebook dùng "Prod. year"
year = st.sidebar.slider(
    "Năm sản xuất",
    1940, 2024, 2018
)

# 7. Các thông số khác (để tăng độ chính xác)
airbags = st.sidebar.slider("Số túi khí", 0, 16, 4)
cylinders = st.sidebar.slider("Số xi-lanh", 1, 16, 4)

if st.button("🔮 Dự đoán giá xe"):
    # Tạo dictionary khớp với tên cột ban đầu của X_train
    data = {
        "Prod. year": year,
        "Engine volume": engine,
        "Mileage": mileage_val,
        "Cylinders": cylinders,
        "Airbags": airbags,
        f"Manufacturer_{manufacturer}": 1,
        f"Fuel type_{fuel_type}": 1,
        f"Gear box type_{gear_box}": 1
    }

    input_df = pd.DataFrame([data])

    # Bổ sung các cột thiếu (giá trị 0) và sắp xếp đúng thứ tự
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Đảm bảo thứ tự cột y hệt như lúc train
    input_df = input_df[columns]

    # Dự đoán (Không dùng expm1 vì notebook dự đoán trực tiếp Price)
    price = model.predict(input_df)[0]

    if price < 0: price = 0 # Tránh giá âm

    st.success(f"💰 Giá xe dự đoán: {price:,.0f} USD")

    with st.expander("Xem chi tiết dữ liệu đầu vào"):
        st.dataframe(input_df)