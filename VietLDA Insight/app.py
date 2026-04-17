import streamlit as st
import re
import json
import os
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from gensim import models, corpora
from collections import Counter

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Phân Tích Chủ Đề",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "lda_model")

# Tên gợi ý cho 30 chủ đề (có thể chỉnh lại sau khi xem từ khóa)
TOPIC_LABELS = {
    i: f"Chủ đề {i + 1}" for i in range(30)
}

# ─────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang tải mô hình LDA...")
def load_model():
    model_path = os.path.join(MODEL_DIR, "lda_model.gensim")
    state_path = model_path + ".state"

    # Kiểm tra file state tồn tại
    if not os.path.exists(state_path):
        st.error(f"❌ Thiếu file: `{state_path}`\n\nVui lòng đảm bảo thư mục `lda_model/` có đủ:\n- lda_model.gensim\n- lda_model.gensim.state\n- lda_model.gensim.id2word\n- lda_model.gensim.expElogbeta.npy")
        st.stop()

    # Load với mmap để đọc đúng file .state và .expElogbeta.npy
    lda = models.LdaModel.load(model_path, mmap='r')

    # Kiểm tra state sau khi load
    if lda.state is None or not hasattr(lda.state, 'get_lambda'):
        st.error("❌ Model load thất bại: `state` bị None. Hãy thử xóa __pycache__ và restart Streamlit.")
        st.stop()

    dic = corpora.Dictionary.load(os.path.join(MODEL_DIR, "dictionary.gensim"))
    sw_path = os.path.join(MODEL_DIR, "stopwords.txt")
    stopwords = set()
    if os.path.exists(sw_path):
        with open(sw_path, encoding="utf-8") as f:
            stopwords = {line.strip() for line in f if line.strip()}
    with open(os.path.join(MODEL_DIR, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return lda, dic, stopwords, meta


# ─────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────
def preprocess_text(text: str, stopwords: set) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(
        r"[^a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơ"
        r"ƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểế\s]",
        " ",
        text,
    )
    try:
        from underthesea import word_tokenize
        tokens = word_tokenize(text, format="text").split()
    except Exception:
        tokens = text.split()
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def predict_topics(text: str, lda, dic, stopwords, top_n: int = 5):
    tokens = preprocess_text(text, stopwords)
    if not tokens:
        return [], tokens
    bow = dic.doc2bow(tokens)
    topics = lda.get_document_topics(bow, minimum_probability=0.0)
    return sorted(topics, key=lambda x: x[1], reverse=True)[:top_n], tokens


def get_topic_keywords(lda, topic_id: int, topn: int = 15):
    words = lda.show_topic(topic_id, topn=topn)
    return words  # list of (word, prob)


# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1923 0%, #1a2940 100%);
}
[data-testid="stSidebar"] * {
    color: #c8d8e8 !important;
}

/* Main bg */
.stApp {
    background: #f5f7fa;
}

/* Hero */
.hero-box {
    background: linear-gradient(135deg, #0f2544 0%, #1e4d8c 50%, #0f3d6e 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-box::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(100,180,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #7eb8f0;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 400;
}

/* Topic card */
.topic-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    border-left: 5px solid #1e6fc4;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    transition: transform 0.15s;
}
.topic-card:hover { transform: translateX(3px); }
.topic-rank { font-family: 'Space Mono', monospace; color: #1e6fc4; font-size: 0.8rem; }
.topic-name { font-weight: 600; font-size: 1.05rem; color: #1a2b45; margin: 0.2rem 0; }
.topic-prob { font-size: 0.85rem; color: #557799; }

/* Metric tile */
.metric-tile {
    background: white;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
.metric-val { font-family: 'Space Mono', monospace; font-size: 1.8rem; color: #1e6fc4; font-weight: 700; }
.metric-lbl { font-size: 0.78rem; color: #7a90a8; text-transform: uppercase; letter-spacing: 0.08em; }

/* Section header */
.section-hdr {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8099b5;
    margin: 1.5rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-hdr::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #dce6f0;
}

/* Token chips */
.token-wrap { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.token-chip {
    background: #e8f0fa;
    color: #1e4d8c;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-family: 'Space Mono', monospace;
}

/* Text area */
.stTextArea textarea {
    border-radius: 10px !important;
    border: 1.5px solid #c8d8ec !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #1e6fc4 !important;
    box-shadow: 0 0 0 3px rgba(30,111,196,0.12) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1e6fc4, #1553a0) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(30,111,196,0.4) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────
lda_model, dictionary, stopwords, metadata = load_model()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt")
    top_n = st.slider("Số chủ đề hiển thị", 1, 10, 5)
    show_tokens = st.checkbox("Hiển thị tokens sau xử lý", value=True)
    kw_per_topic = st.slider("Từ khóa mỗi chủ đề", 5, 20, 10)

    st.markdown("---")
    st.markdown("### 📊 Thông tin mô hình")
    st.markdown(f"- **Chủ đề:** {metadata['num_topics']}")
    st.markdown(f"- **Từ vựng:** {metadata['vocabulary_size']:,}")
    st.markdown(f"- **Tài liệu:** {metadata['num_documents']:,}")
    st.markdown(f"- **Coherence:** {metadata['coherence_score']:.4f}")
    st.markdown(f"- **Passes:** {metadata['passes']}")

    st.markdown("---")
    st.markdown("### 🧭 Điều hướng")
    page = st.radio(
        "",
        ["🔍 Phân tích văn bản", "📚 Khám phá chủ đề", "📊 So sánh nhiều văn bản"],
        label_visibility="collapsed",
    )

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown(
    """
<div class="hero-box">
  <p class="hero-title">🔍 Phân Tích Chủ Đề Văn Bản</p>
  <p class="hero-sub">Mô hình LDA · 30 chủ đề · Tiếng Việt · Trained trên 150,000 tài liệu</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
# PAGE 1: PHÂN TÍCH VĂN BẢN
# ─────────────────────────────────────────
if page == "🔍 Phân tích văn bản":
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-hdr">📝 Nhập văn bản</div>', unsafe_allow_html=True)
        sample_texts = {
            "— Chọn mẫu —": "",
            "Bóng đá": "Đội tuyển Việt Nam vừa giành chiến thắng 2-0 trước Thái Lan trong trận đấu AFF Cup. Huấn luyện viên Park Hang-seo khen ngợi các cầu thủ đã thi đấu xuất sắc và đoàn kết.",
            "Kinh tế": "Ngân hàng Nhà nước Việt Nam vừa điều chỉnh lãi suất cơ bản nhằm kiểm soát lạm phát. Các chuyên gia kinh tế nhận định GDP quý III có thể đạt 6,5% tăng trưởng.",
            "Công nghệ": "Trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc và học tập. Các công ty công nghệ lớn đua nhau đầu tư vào nghiên cứu và phát triển mô hình ngôn ngữ lớn.",
            "Y tế": "Bộ Y tế khuyến cáo người dân tiêm vắc xin phòng cúm mùa đông. Các bệnh viện tăng cường giám sát dịch bệnh và chuẩn bị kịch bản ứng phó với ca bệnh mới.",
        }
        selected = st.selectbox("Văn bản mẫu", list(sample_texts.keys()))
        default_val = sample_texts[selected]

        user_text = st.text_area(
            "Nhập văn bản tiếng Việt",
            value=default_val,
            height=220,
            placeholder="Dán hoặc nhập văn bản cần phân tích tại đây...",
            label_visibility="collapsed",
        )
        analyze_btn = st.button("🚀 Phân tích ngay", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-hdr">🎯 Kết quả phân tích</div>', unsafe_allow_html=True)

        if analyze_btn and user_text.strip():
            with st.spinner("Đang phân tích..."):
                topics, tokens = predict_topics(user_text, lda_model, dictionary, stopwords, top_n)

            if not topics:
                st.warning("Không nhận diện được chủ đề. Văn bản có thể quá ngắn hoặc toàn stopword.")
            else:
                # Metrics row
                top_prob = topics[0][1]
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f'<div class="metric-tile"><div class="metric-val">{len(topics)}</div><div class="metric-lbl">Chủ đề tìm thấy</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-tile"><div class="metric-val">{len(tokens)}</div><div class="metric-lbl">Tokens hữu ích</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-tile"><div class="metric-val">{top_prob:.0%}</div><div class="metric-lbl">Độ tin cậy cao nhất</div></div>', unsafe_allow_html=True)

                st.markdown("")

                # Topic cards
                colors = ["#1e6fc4", "#2a8f5c", "#c45c1e", "#8c1ec4", "#c41e5c",
                          "#1ea8c4", "#a8c41e", "#c4a81e", "#5c2ac4", "#c42a5c"]
                for rank, (tid, prob) in enumerate(topics):
                    color = colors[rank % len(colors)]
                    kws = get_topic_keywords(lda_model, tid, topn=5)
                    kw_str = " · ".join([w for w, _ in kws])
                    st.markdown(
                        f"""
<div class="topic-card" style="border-left-color:{color}">
  <div class="topic-rank">#{rank+1} / TOPIC {tid+1}</div>
  <div class="topic-name">{TOPIC_LABELS.get(tid, f'Chủ đề {tid+1}')}</div>
  <div class="topic-prob">Xác suất: <strong>{prob:.2%}</strong></div>
  <div style="font-size:0.78rem;color:#8099b5;margin-top:0.3rem">{kw_str}</div>
</div>""",
                        unsafe_allow_html=True,
                    )

                # Bar chart — cải tiến
                st.markdown('<div class="section-hdr">📈 Biểu đồ phân bố</div>', unsafe_allow_html=True)
                df_topics = pd.DataFrame(topics, columns=["topic_id", "prob"])
                df_topics["label"] = df_topics["topic_id"].apply(lambda x: f"Topic {x+1}")
                df_topics["pct"] = df_topics["prob"].apply(lambda x: f"{x:.1%}")
                df_topics["kw"] = df_topics["topic_id"].apply(
                    lambda x: " · ".join([w for w, _ in get_topic_keywords(lda_model, x, topn=3)])
                )
                colors_bar = ["#1e6fc4","#2a8f5c","#c45c1e","#8c1ec4","#c41e5c",
                              "#1ea8c4","#a8c41e","#c4a81e","#5c2ac4","#c42a5c"]
                max_prob = df_topics["prob"].max()
                df_topics["norm"] = df_topics["prob"] / max_prob * 100
                fig = go.Figure()
                for i, row in df_topics.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row["norm"]],
                        y=[row["label"]],
                        orientation="h",
                        marker=dict(color=colors_bar[i % len(colors_bar)], line=dict(color="rgba(255,255,255,0.3)", width=1)),
                        text=f"  {row['pct']}  {row['kw']}",
                        textposition="inside",
                        textfont=dict(color="white", size=11, family="Be Vietnam Pro"),
                        hovertemplate=f"<b>{row['label']}</b><br>Xac suat: {row['pct']}<br>{row['kw']}<extra></extra>",
                        showlegend=False,
                    ))
                fig.update_layout(
                    height=max(200, len(topics) * 44 + 40),
                    margin=dict(l=10, r=20, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(248,251,255,0.6)",
                    font=dict(family="Be Vietnam Pro, sans-serif", size=12),
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 115]),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=12, color="#1a2b45", family="Space Mono, monospace")),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Tokens
                if show_tokens and tokens:
                    st.markdown('<div class="section-hdr">🔤 Tokens sau xử lý</div>', unsafe_allow_html=True)
                    chips = "".join([f'<span class="token-chip">{t}</span>' for t in tokens[:40]])
                    st.markdown(f'<div class="token-wrap">{chips}</div>', unsafe_allow_html=True)

        elif analyze_btn:
            st.info("Vui lòng nhập văn bản trước khi phân tích.")
        else:
            st.markdown(
                """
<div style="text-align:center;padding:3rem 1rem;color:#8099b5;">
  <div style="font-size:3rem">📄</div>
  <div style="margin-top:0.5rem;font-size:0.95rem">Nhập văn bản và nhấn <strong>Phân tích ngay</strong></div>
</div>""",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────
# PAGE 2: KHÁM PHÁ CHỦ ĐỀ
# ─────────────────────────────────────────
elif page == "📚 Khám phá chủ đề":
    st.markdown('<div class="section-hdr">📚 Từ khóa theo chủ đề</div>', unsafe_allow_html=True)

    col_sel, col_viz = st.columns([1, 2], gap="large")

    with col_sel:
        selected_topic = st.selectbox(
            "Chọn chủ đề để khám phá",
            options=list(range(metadata["num_topics"])),
            format_func=lambda x: f"Topic {x+1} — {TOPIC_LABELS.get(x, '')}",
        )
        words = get_topic_keywords(lda_model, selected_topic, topn=kw_per_topic)

        st.markdown(f"**Topic {selected_topic + 1}** · Top {kw_per_topic} từ khóa")
        for w, p in words:
            bar_w = int(p / words[0][1] * 100)
            st.markdown(
                f"""
<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.4rem">
  <span style="font-family:'Space Mono',monospace;font-size:0.82rem;width:120px;color:#1a2b45">{w}</span>
  <div style="flex:1;background:#e8f0fa;border-radius:4px;height:8px">
    <div style="width:{bar_w}%;background:#1e6fc4;height:8px;border-radius:4px"></div>
  </div>
  <span style="font-size:0.75rem;color:#8099b5;width:50px;text-align:right">{p:.4f}</span>
</div>""",
                unsafe_allow_html=True,
            )

    with col_viz:
        # Bubble chart of keyword weights
        df_kw = pd.DataFrame(words, columns=["word", "weight"])
        fig2 = go.Figure()
        for i, row in df_kw.iterrows():
            fig2.add_trace(
                go.Scatter(
                    x=[i],
                    y=[row["weight"]],
                    mode="markers+text",
                    marker=dict(
                        size=row["weight"] / df_kw["weight"].max() * 60 + 20,
                        color=f"rgba(30,111,196,{0.3 + 0.7 * row['weight'] / df_kw['weight'].max()})",
                        line=dict(color="#1e6fc4", width=1.5),
                    ),
                    text=row["word"],
                    textposition="middle center",
                    textfont=dict(size=10, color="white", family="Be Vietnam Pro"),
                    hovertemplate=f"<b>{row['word']}</b><br>Weight: {row['weight']:.4f}<extra></extra>",
                    showlegend=False,
                )
            )
        fig2.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(title="Weight", showgrid=True, gridcolor="#e8f0fa"),
            font=dict(family="Be Vietnam Pro, sans-serif"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Topic heatmap
    st.markdown('<div class="section-hdr">🗺️ Tổng quan tất cả chủ đề (Top 10 từ)</div>', unsafe_allow_html=True)
    all_words_set = set()
    topic_word_matrix = {}
    for tid in range(min(30, metadata["num_topics"])):
        kws = dict(get_topic_keywords(lda_model, tid, topn=10))
        topic_word_matrix[tid] = kws
        all_words_set.update(kws.keys())

    top_global_words = list(all_words_set)[:30]
    matrix_data = []
    for tid in range(min(30, metadata["num_topics"])):
        row = [topic_word_matrix[tid].get(w, 0) for w in top_global_words]
        matrix_data.append(row)

    fig3 = go.Figure(
        go.Heatmap(
            z=matrix_data,
            x=top_global_words,
            y=[f"T{i+1}" for i in range(len(matrix_data))],
            colorscale=[[0, "#f0f6ff"], [1, "#1e4d8c"]],
            hovertemplate="Topic %{y} · %{x}<br>Weight: %{z:.4f}<extra></extra>",
        )
    )
    fig3.update_layout(
        height=500,
        margin=dict(l=40, r=20, t=20, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Be Vietnam Pro, sans-serif", size=11),
        xaxis=dict(tickangle=-35),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 3: SO SÁNH NHIỀU VĂN BẢN
# ─────────────────────────────────────────
elif page == "📊 So sánh nhiều văn bản":
    st.markdown('<div class="section-hdr">📊 So sánh phân bố chủ đề của nhiều văn bản</div>', unsafe_allow_html=True)

    n_docs = st.number_input("Số văn bản cần so sánh", min_value=2, max_value=6, value=3)

    docs = []
    cols = st.columns(min(n_docs, 3))
    for i in range(n_docs):
        c = cols[i % 3]
        with c:
            t = st.text_area(f"Văn bản {i+1}", height=140, key=f"doc_{i}",
                             placeholder=f"Nhập văn bản {i+1}...")
            docs.append(t)

    if st.button("🔄 So sánh tất cả", use_container_width=True):
        results = []
        for i, doc in enumerate(docs):
            if doc.strip():
                topics, _ = predict_topics(doc, lda_model, dictionary, stopwords, top_n=metadata["num_topics"])
                dist = {tid: 0.0 for tid in range(metadata["num_topics"])}
                for tid, prob in topics:
                    dist[tid] = prob
                results.append((f"Văn bản {i+1}", dist))

        if len(results) < 2:
            st.warning("Cần ít nhất 2 văn bản có nội dung.")
        else:
            # Radar chart for top-10 topics
            top_topics_union = set()
            for _, dist in results:
                top_10 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:10]
                top_topics_union.update([t for t, _ in top_10])
            top_topics_union = sorted(top_topics_union)
            labels = [f"T{t+1}" for t in top_topics_union]

            fig4 = go.Figure()
            palette = ["#1e6fc4", "#2a8f5c", "#c45c1e", "#8c1ec4", "#c41e5c", "#1ea8c4"]
            for idx, (name, dist) in enumerate(results):
                vals = [dist.get(t, 0) for t in top_topics_union]
                fig4.add_trace(
                    go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=labels + [labels[0]],
                        fill="toself",
                        name=name,
                        line=dict(color=palette[idx % len(palette)], width=2),
                        fillcolor=palette[idx % len(palette)].replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in palette[idx % len(palette)] else palette[idx % len(palette)] + "26",
                        opacity=0.85,
                    )
                )
            fig4.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max(
                        max(dist.values()) for _, dist in results
                    ) * 1.1]),
                    bgcolor="rgba(240,246,255,0.5)",
                ),
                height=450,
                margin=dict(l=60, r=60, t=40, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Be Vietnam Pro, sans-serif"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            )
            st.plotly_chart(fig4, use_container_width=True)

            # Stacked bar
            st.markdown('<div class="section-hdr">📊 Phân bố xếp chồng (Top 10 topics)</div>', unsafe_allow_html=True)
            df_compare = pd.DataFrame(
                {name: [dist.get(t, 0) for t in top_topics_union] for name, dist in results},
                index=labels,
            )
            fig5 = go.Figure()
            for col in df_compare.columns:
                fig5.add_trace(go.Bar(name=col, x=df_compare.index, y=df_compare[col]))
            fig5.update_layout(
                barmode="group",
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Be Vietnam Pro, sans-serif"),
                legend=dict(orientation="h"),
                xaxis=dict(title="Chủ đề"),
                yaxis=dict(title="Xác suất"),
            )
            st.plotly_chart(fig5, use_container_width=True)