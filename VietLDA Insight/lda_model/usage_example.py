
# ============ CÁCH SỬ DỤNG MÔ HÌNH LDA ============
from gensim import models, corpora
import re
from underthesea import word_tokenize

# Load model
lda_model = models.LdaModel.load('lda_model.gensim')
dictionary = corpora.Dictionary.load('dictionary.gensim')

# Stopwords (đã dùng khi training)
STOPWORDS = set([line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8')])

def preprocess_text(text):
    """Tiền xử lý văn bản"""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểế\s]', ' ', text)
    try:
        tokens = word_tokenize(text, format="text").split()
    except:
        tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def predict_topic(text, top_n=3):
    """Dự đoán chủ đề cho văn bản"""
    tokens = preprocess_text(text)
    if not tokens:
        return []
    bow = dictionary.doc2bow(tokens)
    topics = lda_model.get_document_topics(bow)
    return sorted(topics, key=lambda x: x[1], reverse=True)[:top_n]

# Ví dụ sử dụng
if __name__ == "__main__":
    text = "Đội tuyển Việt Nam giành chiến thắng"
    topics = predict_topic(text)
    print(f"Text: {text}")
    for topic_id, prob in topics:
        print(f"  Topic {topic_id+1}: {prob:.2%}")
