# 🔍 Phân Tích Chủ Đề Văn Bản (LDA Streamlit App)

## Cấu trúc thư mục

```
project/
├── app.py
├── requirements.txt
└── lda_model/
    ├── lda_model.gensim
    ├── dictionary.gensim
    ├── metadata.json
    └── stopwords.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

## Tính năng

### 🔍 Phân tích văn bản
- Nhập văn bản tiếng Việt bất kỳ
- Hiển thị top N chủ đề có xác suất cao nhất
- Biểu đồ bar chart phân bố chủ đề
- Xem tokens sau khi tiền xử lý

### 📚 Khám phá chủ đề
- Xem từ khóa của từng chủ đề (Top 5–20 từ)
- Bubble chart trọng số từ khóa
- Heatmap tổng quan 30 chủ đề

### 📊 So sánh nhiều văn bản
- So sánh 2–6 văn bản cùng lúc
- Radar chart phân bố chủ đề
- Biểu đồ grouped bar
