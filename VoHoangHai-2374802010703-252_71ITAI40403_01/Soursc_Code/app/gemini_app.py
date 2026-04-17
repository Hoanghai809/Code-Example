# gemini_app.py
# Ứng dụng demo mô hình Gemini cho gợi ý nhạc - CÓ KHUNG CHAT

import streamlit as st
import torch
import pickle
import pandas as pd
import numpy as np
import os
import sys
import re
from datetime import datetime

# Thêm đường dẫn để import model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_architecture import GeminiMusicModel

# ========== CẤU HÌNH TRANG ==========
st.set_page_config(
    page_title="Gemini Music Assistant",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CSS ==========
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        padding: 1rem;
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .main-header h1 {
        font-size: 2rem;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.9);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 0.8rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background: rgba(255,255,255,0.95);
        color: #333;
        margin-right: 20%;
        border-left: 4px solid #667eea;
    }
    .chat-avatar {
        font-size: 1.2rem;
        margin-right: 0.8rem;
    }
    .chat-content {
        flex: 1;
    }
    .timestamp {
        font-size: 0.6rem;
        color: #999;
        margin-top: 5px;
    }
    
    /* Song card styling */
    .song-card {
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        padding: 10px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .song-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .song-title {
        font-size: 0.95rem;
        font-weight: bold;
        color: #333;
    }
    .song-artist {
        font-size: 0.8rem;
        color: #666;
    }
    .song-genre {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2px 8px;
        border-radius: 15px;
        font-size: 0.7rem;
    }
    .score-high { color: #4CAF50; font-weight: bold; }
    .score-medium { color: #FF9800; font-weight: bold; }
    .score-low { color: #f44336; font-weight: bold; }
    
    .metric-card {
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        padding: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .footer {
        text-align: center;
        padding: 15px;
        color: rgba(255,255,255,0.7);
        font-size: 0.7rem;
        margin-top: 20px;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: bold;
        width: 100%;
    }
    .stButton button:hover {
        transform: scale(1.02);
    }
    
    /* Fix cho text input */
    .stTextInput input {
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ========== ĐỊNH NGHĨA ĐƯỜNG DẪN ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')


# ========== HÀM LOAD MODEL ==========
@st.cache_resource
def load_model_and_artifacts():
    """Load model, encoders, scaler và database"""
    
    # Load encoders
    with open(os.path.join(MODEL_DIR, 'genre_encoder.pkl'), 'rb') as f:
        genre_encoder = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, 'artist_encoder.pkl'), 'rb') as f:
        artist_encoder = pickle.load(f)
    
    # Load scaler
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load songs database
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_7k_metadata_authors.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_1k_metadata_authors.csv'))
    songs_df = pd.concat([train_df, val_df], ignore_index=True)
    songs_df = songs_df[['title', 'artist', 'genre', 'link']].copy()
    songs_df.columns = ['title', 'artist', 'genre', 'zing_link']
    
    # Load model
    num_genres = len(genre_encoder.classes_)
    num_artists = len(artist_encoder.classes_)
    
    model = GeminiMusicModel(
        num_genres=num_genres,
        num_artists=num_artists,
        embed_dim=128,
        num_heads=8,
        num_layers=6,
        ff_dim=512,
        dropout=0.2
    )
    
    # Load weights
    model_path = os.path.join(MODEL_DIR, 'ultra_best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, 'best_model.pth')
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model, genre_encoder, artist_encoder, scaler, songs_df


# ========== HÀM XỬ LÝ CHAT ==========
def parse_user_intent(message, genre_encoder, artist_encoder):
    """Phân tích ý định người dùng từ tin nhắn chat"""
    message_lower = message.lower().strip()
    
    # Mặc định
    result = {
        'genre': None,
        'artist': None,
        'mood': None,
        'action': 'recommend'
    }
    
    # Từ khóa thể loại
    genre_keywords = {
        'pop': 'Pop',
        'ballad': 'Ballad',
        'rap': 'Rap',
        'edm': 'EDM',
        'rock': 'Rock',
        'indie': 'Indie',
        'lofi': 'Lofi',
        'nhạc trẻ': 'Nhạc trẻ',
        'v-pop': 'V-Pop'
    }
    
    # Từ khóa tâm trạng -> thể loại
    mood_keywords = {
        'buồn': 'Ballad',
        'cô đơn': 'Ballad',
        'vui': 'Pop',
        'hạnh phúc': 'Pop',
        'năng động': 'EDM',
        'tập gym': 'EDM',
        'học tập': 'Lofi',
        'làm việc': 'Lofi',
        'chill': 'Indie'
    }
    
    # Tìm thể loại
    for key, val in genre_keywords.items():
        if key in message_lower:
            result['genre'] = val
            break
    
    # Tìm tâm trạng
    for key, val in mood_keywords.items():
        if key in message_lower:
            result['mood'] = key
            result['genre'] = val
            break
    
    # Tìm ca sĩ (kiểm tra trong encoder)
    all_artists = list(artist_encoder.classes_)
    for artist in all_artists[:100]:
        if artist.lower() in message_lower:
            result['artist'] = artist
            break
    
    # Xác định action
    if 'giới thiệu' in message_lower or 'giới thiệu về' in message_lower:
        result['action'] = 'info'
    elif 'cảm ơn' in message_lower or 'thanks' in message_lower:
        result['action'] = 'thank'
    elif 'chào' in message_lower or 'hello' in message_lower or 'hi' in message_lower:
        result['action'] = 'greet'
    
    return result


def get_gemini_response(intent, genre_encoder, artist_encoder):
    """Tạo phản hồi dạng text từ Gemini"""
    
    if intent['action'] == 'greet':
        return "🎵 Xin chào! Tôi là Gemini Music Assistant. Tôi có thể giúp bạn gợi ý nhạc theo thể loại, ca sĩ hoặc tâm trạng. Bạn muốn nghe nhạc gì hôm nay?"
    
    elif intent['action'] == 'thank':
        return "😊 Rất vui được giúp bạn! Nếu cần thêm gợi ý, hãy nói với tôi nhé. Chúc bạn nghe nhạc vui vẻ!"
    
    elif intent['action'] == 'info':
        return """🎵 **Giới thiệu về Gemini Music Assistant**

Tôi là trợ lý âm nhạc thông minh được xây dựng dựa trên kiến trúc **Google Gemini**.

**Các tính năng:**
- 🎼 Gợi ý nhạc theo thể loại
- 🎤 Gợi ý nhạc theo ca sĩ  
- 😊 Gợi ý theo tâm trạng (vui, buồn, năng động...)
- 🎧 Nghe nhạc trực tiếp trên Zing MP3

**Hãy thử nói:**
- "Tôi muốn nghe nhạc buồn"
- "Gợi ý nhạc Pop"
- "Bài hát của Sơn Tùng"
"""
    
    else:
        if intent['genre'] and intent['artist']:
            return f"🎵 Tôi sẽ gợi ý cho bạn những bài hát **{intent['genre']}** của ca sĩ **{intent['artist']}** và các bài hát cùng thể loại nhé!"
        elif intent['genre']:
            return f"🎵 Tôi sẽ gợi ý cho bạn những bài hát **{intent['genre']}** hay nhất!"
        elif intent['artist']:
            return f"🎤 Tôi sẽ gợi ý cho bạn những bài hát của ca sĩ **{intent['artist']}** và các bài hát cùng thể loại!"
        elif intent['mood']:
            return f"😊 Bạn đang {intent['mood']}? Tôi sẽ gợi ý nhạc **{intent['genre']}** phù hợp với tâm trạng này!"
        else:
            return "🎵 Tôi có thể gợi ý nhạc cho bạn! Hãy cho tôi biết bạn thích thể loại nào (Pop, Ballad, Rap, EDM...), ca sĩ nào, hoặc tâm trạng hiện tại của bạn nhé!"


# ========== HÀM GỢI Ý (ĐÃ SỬA LỖI LẶP) ==========
def recommend_songs(model, genre_encoder, artist_encoder, scaler, songs_df,
                    user_genre=None, user_artist=None, top_k=10):
    """Gợi ý bài hát dựa trên sở thích - Đã sửa lỗi lặp"""
    
    recommendations = []
    
    # Lọc theo thể loại trước
    filtered_df = songs_df.copy()
    
    if user_genre and user_genre != "Tất cả":
        # Lọc đúng thể loại
        filtered_df = filtered_df[filtered_df['genre'].str.contains(user_genre, case=False, na=False)]
        
        # Nếu không có bài hát nào, thử tìm gần đúng
        if len(filtered_df) == 0:
            # Tìm các bài có genre chứa từ khóa
            for genre in songs_df['genre'].unique():
                if user_genre.lower() in str(genre).lower():
                    filtered_df = songs_df[songs_df['genre'] == genre]
                    break
    
    if user_artist and user_artist != "Tất cả":
        filtered_df = filtered_df[filtered_df['artist'].str.contains(user_artist, case=False, na=False)]
    
    # Nếu không có kết quả, lấy tất cả
    if len(filtered_df) == 0:
        filtered_df = songs_df
    
    # Giới hạn số lượng để tránh quá tải
    filtered_df = filtered_df.head(200)
    
    # Encode user preferences
    if user_genre and user_genre != "Tất cả":
        try:
            user_genre_id = genre_encoder.transform([user_genre])[0]
        except:
            user_genre_id = None
    else:
        user_genre_id = None
    
    if user_artist and user_artist != "Tất cả":
        try:
            user_artist_id = artist_encoder.transform([user_artist])[0]
        except:
            user_artist_id = None
    else:
        user_artist_id = None
    
    with torch.no_grad():
        for idx, song in filtered_df.iterrows():
            song_genre = song['genre']
            song_artist = song['artist']
            
            if pd.isna(song_genre) or pd.isna(song_artist):
                continue
            
            try:
                song_genre_id = genre_encoder.transform([song_genre])[0]
                song_artist_id = artist_encoder.transform([song_artist])[0]
            except:
                continue
            
            final_user_genre = user_genre_id if user_genre_id is not None else song_genre_id
            final_user_artist = user_artist_id if user_artist_id is not None else song_artist_id
            
            # Dự đoán
            inputs = (
                torch.tensor([song_genre_id], dtype=torch.long),
                torch.tensor([song_artist_id], dtype=torch.long),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([final_user_genre], dtype=torch.long),
                torch.tensor([final_user_artist], dtype=torch.long)
            )
            
            score = model(*inputs).item()
            
            # Chỉ lấy bài có score > 0.5 để tránh gợi ý tầm bậy
            if score > 0.5:
                recommendations.append({
                    'title': song['title'],
                    'artist': song['artist'],
                    'genre': song_genre,
                    'score': score,
                    'zing_link': song.get('zing_link', None)
                })
    
    # Loại bỏ trùng lặp theo title
    seen_titles = set()
    unique_recs = []
    for rec in recommendations:
        if rec['title'] not in seen_titles:
            seen_titles.add(rec['title'])
            unique_recs.append(rec)
    
    recommendations = unique_recs
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    # Đảm bảo không trả về bài hát trùng
    return recommendations[:top_k]

# ========== KHỞI TẠO SESSION STATE ==========
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "🎵 Xin chào! Tôi là Gemini Music Assistant. Tôi có thể giúp bạn gợi ý nhạc theo thể loại, ca sĩ hoặc tâm trạng. Bạn muốn nghe nhạc gì hôm nay?", "timestamp": datetime.now()}
    ]

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

if 'last_input' not in st.session_state:
    st.session_state.last_input = ""

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = [
        {
            'id': 1,
            'name': 'Cuộc trò chuyện 1',
            'messages': [
                {"role": "assistant", "content": "🎵 Xin chào! Tôi là Gemini Music Assistant. Tôi có thể giúp bạn gợi ý nhạc theo thể loại, ca sĩ hoặc tâm trạng. Bạn muốn nghe nhạc gì hôm nay?", "timestamp": datetime.now()}
            ],
            'recommendations': None
        }
    ]

if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = 1


# ========== GIAO DIỆN CHÍNH ==========
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Gemini Music Assistant</h1>
        <p>Trợ lý âm nhạc thông minh - Mô phỏng kiến trúc Google Gemini</p>
        <p style="font-size: 0.8rem;">🎤 Hỗ trợ nhạc Việt | 🎧 Nghe trực tiếp | 💬 Chat để được gợi ý</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Đang khởi động Gemini Music Assistant..."):
        model, genre_encoder, artist_encoder, scaler, songs_df = load_model_and_artifacts()
    
    # ========== SIDEBAR GIỐNG GEMINI ==========
    with st.sidebar:
        # Header sidebar
        st.markdown("""
        <div style="text-align: center; padding: 10px; margin-bottom: 20px;">
            <h2 style="color: #667eea; margin: 0;">✨ Gemini</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Nút "Cuộc trò chuyện mới"
        if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
            new_chat_id = len(st.session_state.chat_sessions) + 1
            st.session_state.chat_sessions.append({
                'id': new_chat_id,
                'name': f"Cuộc trò chuyện {new_chat_id}",
                'messages': [
                    {"role": "assistant", "content": "🎵 Xin chào! Tôi là Gemini Music Assistant. Tôi có thể giúp bạn gợi ý nhạc theo thể loại, ca sĩ hoặc tâm trạng. Bạn muốn nghe nhạc gì hôm nay?", "timestamp": datetime.now()}
                ],
                'recommendations': None
            })
            st.session_state.current_chat_id = new_chat_id
            st.session_state.messages = st.session_state.chat_sessions[-1]['messages']
            st.session_state.recommendations = None
            st.rerun()
        
        st.markdown("---")
        
        # Mục "Nội dung của tôi"
        st.markdown("""
        <div style="margin: 10px 0;">
            <p style="color: #666; font-size: 14px; margin: 5px 0;">📁 Nội dung của tôi</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mục "Gem"
        with st.expander("✨ Gem", expanded=False):
            st.markdown("""
            - 🎵 **Gợi ý nhạc theo thể loại**
            - 🎤 **Gợi ý theo ca sĩ**
            - 😊 **Gợi ý theo tâm trạng**
            - 📊 **Phân tích xu hướng nhạc**
            """)
        
        st.markdown("---")
        
        # Lịch sử cuộc trò chuyện
        st.markdown("### 📜 Lịch sử")
        
        for idx, chat in enumerate(st.session_state.chat_sessions):
            first_user_msg = ""
            for msg in chat['messages']:
                if msg['role'] == 'user':
                    first_user_msg = msg['content'][:30] + ("..." if len(msg['content']) > 30 else "")
                    break
            
            if not first_user_msg:
                first_user_msg = f"Cuộc trò chuyện {chat['id']}"
            
            unique_key = f"{chat['id']}_{idx}"
            
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(f"💬 {first_user_msg}", key=f"chat_{unique_key}", use_container_width=True):
                    st.session_state.current_chat_id = chat['id']
                    st.session_state.messages = chat['messages']
                    st.session_state.recommendations = chat.get('recommendations', None)
                    st.rerun()
         
        
        # Phần sở thích
        with st.expander("🎧 Sở thích của bạn", expanded=False):
            genres = sorted(['Tất cả'] + list(genre_encoder.classes_))
            artists = sorted(['Tất cả'] + list(artist_encoder.classes_)[:50])
            
            selected_genre = st.selectbox("🎼 Thể loại yêu thích", genres)
            selected_artist = st.selectbox("🎤 Ca sĩ yêu thích", artists)
            
            mood_map = {
                "🎉 Vui vẻ": "Pop",
                "😢 Buồn": "Ballad",
                "💪 Năng động": "EDM",
                "📚 Học tập": "Lofi",
                "🎸 Chill": "Indie"
            }
            
            selected_mood = st.selectbox("😊 Tâm trạng", ["Chọn..."] + list(mood_map.keys()))
            if selected_mood != "Chọn...":
                selected_genre = mood_map[selected_mood]
                st.success(f"✅ Gợi ý: {selected_genre}")
            
            if st.button("🔍 GỢI Ý NGAY", use_container_width=True):
                with st.spinner("🎵 Đang tìm kiếm..."):
                    recommendations = recommend_songs(
                        model, genre_encoder, artist_encoder, scaler, songs_df,
                        selected_genre if selected_genre != "Tất cả" else None,
                        selected_artist if selected_artist != "Tất cả" else None,
                        10
                    )
                    st.session_state.recommendations = recommendations
                    for chat in st.session_state.chat_sessions:
                        if chat['id'] == st.session_state.current_chat_id:
                            chat['recommendations'] = recommendations
                            break
                    
                    if recommendations:
                        song_list = ""
                        for i, song in enumerate(recommendations[:5], 1):
                            song_list += f"\n\n{i}. **{song['title']}** - {song['artist']} (⭐ {song['score']*100:.1f}%)"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"🔍 Đã tìm thấy {len(recommendations)} bài hát phù hợp!{song_list}",
                            "timestamp": datetime.now()
                        })
                    st.rerun()
        
        st.markdown("---")
        
        # Model info
        with st.expander("📊 Thông tin model", expanded=False):
            st.markdown(f"""
            - **Kiến trúc:** Transformer
            - **Attention Heads:** 8
            - **Layers:** 6
            - **Parameters:** 2M
            - **Accuracy:** 99.66%
            - **Genres:** {len(genre_encoder.classes_)}
            - **Artists:** {len(artist_encoder.classes_)}
            """)
    
    # ========== KHUNG CHAT ==========
    st.subheader("💬 Chat với Gemini")
    
    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="chat-avatar">👤</div>
                <div class="chat-content">
                    <strong>Bạn</strong><br>{msg["content"]}
                    <div class="timestamp">{msg["timestamp"].strftime('%H:%M')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="chat-avatar">🤖</div>
                <div class="chat-content">
                    <strong>Gemini</strong><br>{msg["content"]}
                    <div class="timestamp">{msg["timestamp"].strftime('%H:%M')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== INPUT CHAT ==========
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "", 
            placeholder="💬 Bạn muốn nghe nhạc gì?",
            key="chat_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("📤 Gửi", use_container_width=True, key="send_btn")
    
    # Xử lý tin nhắn
    if send_button or (user_input and user_input != st.session_state.get('last_input', '')):
        if user_input:
            st.session_state.last_input = user_input
            
            # Thêm tin nhắn user
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # Phân tích ý định
            intent = parse_user_intent(user_input, genre_encoder, artist_encoder)
            
            # Tạo phản hồi text
            response = get_gemini_response(intent, genre_encoder, artist_encoder)
            
            # Kiểm tra xem có nên gợi ý không
            should_recommend = False
            recommend_genre = None
            recommend_artist = None
            
            if intent['action'] == 'recommend' or intent['genre'] or intent['artist'] or intent['mood']:
                should_recommend = True
                recommend_genre = intent['genre']
                recommend_artist = intent['artist']
                
                if intent['mood'] and not recommend_genre:
                    mood_to_genre = {
                        'buồn': 'Ballad', 'cô đơn': 'Ballad', 'vui': 'Pop',
                        'năng động': 'EDM', 'tập gym': 'EDM', 'chill': 'Indie',
                        'học tập': 'Lofi', 'làm việc': 'Lofi'
                    }
                    recommend_genre = mood_to_genre.get(intent['mood'], None)
            
            # Gợi ý ngay trong câu trả lời
            if should_recommend:
                with st.spinner("🎵 Gemini đang tìm kiếm bài hát phù hợp..."):
                    recommendations = recommend_songs(
                        model, genre_encoder, artist_encoder, scaler, songs_df,
                        recommend_genre, recommend_artist, 5
                    )
                    st.session_state.recommendations = recommendations
                    
                    if recommendations:
                        song_list = ""
                        for i, song in enumerate(recommendations[:5], 1):
                            score_percent = song['score'] * 100
                            song_list += f"\n\n{i}. **{song['title']}** - {song['artist']}  \n   🎼 {song['genre']} | ⭐ {score_percent:.1f}% phù hợp"
                            
                            if song['zing_link'] and not pd.isna(song['zing_link']):
                                song_list += f"\n   🎧 [Nghe trên Zing MP3]({song['zing_link']})"
                        
                        full_response = response + f"\n\n---\n### 🎵 **Gợi ý cho bạn:**\n{song_list}\n\n---\n💡 *Nhấn vào link để nghe nhạc trực tiếp!*"
                    else:
                        full_response = response + "\n\n---\n😔 *Xin lỗi, không tìm thấy bài hát phù hợp với yêu cầu của bạn.*"
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.now()
                    })
                    
                    # Cập nhật vào chat session
                    for chat in st.session_state.chat_sessions:
                        if chat['id'] == st.session_state.current_chat_id:
                            chat['messages'] = st.session_state.messages
                            chat['recommendations'] = recommendations
                            break
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now()
                })
                st.session_state.recommendations = None
                
                # Cập nhật vào chat session
                for chat in st.session_state.chat_sessions:
                    if chat['id'] == st.session_state.current_chat_id:
                        chat['messages'] = st.session_state.messages
                        break
            
            st.rerun()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🎵 Gemini Music Assistant | Mô phỏng kiến trúc Google Gemini | © 2024</p>
        <p style="font-size: 0.7rem;">💬 Chat để được gợi ý nhạc | 🎧 Nhấn "Nghe" để mở Zing MP3</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()