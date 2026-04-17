# model_architecture.py
# Định nghĩa kiến trúc Ultra Deep Model (Gemini)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention - Cơ chế attention của Gemini"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block - Kiến trúc cốt lõi của Gemini"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class GeminiMusicModel(nn.Module):
    """
    Mô hình Gemini cho gợi ý nhạc
    Kiến trúc dựa trên Transformer và Multi-Head Attention
    """
    def __init__(self, num_genres, num_artists, 
                 embed_dim=128, num_heads=8, num_layers=6, ff_dim=512, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Embedding layers - Gemini style
        self.genre_emb = nn.Embedding(num_genres, embed_dim)
        self.artist_emb = nn.Embedding(num_artists, embed_dim)
        
        nn.init.xavier_uniform_(self.genre_emb.weight)
        nn.init.xavier_uniform_(self.artist_emb.weight)
        
        # Numerical feature processor
        self.num_processor = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Fusion layers
        self.song_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.user_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer layers (Gemini core)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Projection layers
        self.song_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.user_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Cross attention (Gemini's multi-modal capability)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, song_genre, song_artist, song_wpm, song_duration, song_tokens,
                user_genre, user_artist):
        
        # Embeddings
        song_genre_vec = self.genre_emb(song_genre)
        song_artist_vec = self.artist_emb(song_artist)
        user_genre_vec = self.genre_emb(user_genre)
        user_artist_vec = self.artist_emb(user_artist)
        
        # Numerical features
        num_features = torch.stack([song_wpm, song_duration, song_tokens], dim=1)
        num_vec = self.num_processor(num_features)
        
        # Fusion
        song_features = torch.cat([song_genre_vec, song_artist_vec, num_vec], dim=1)
        song_encoded = self.song_fusion(song_features)
        
        user_features = torch.cat([user_genre_vec, user_artist_vec], dim=1)
        user_encoded = self.user_fusion(user_features)
        
        # Transformer encoding
        sequence = torch.stack([song_encoded, user_encoded], dim=1)
        for transformer in self.transformer_layers:
            sequence = transformer(sequence)
        
        song_transformed = sequence[:, 0, :]
        user_transformed = sequence[:, 1, :]
        
        # Projection
        song_proj = self.song_projection(song_transformed)
        user_proj = self.user_projection(user_transformed)
        
        # Cross attention (Gemini's special feature)
        cross_out, _ = self.cross_attention(
            user_proj.unsqueeze(1),
            song_proj.unsqueeze(1),
            song_proj.unsqueeze(1)
        )
        cross_out = cross_out.squeeze(1)
        cross_out = self.cross_norm(cross_out + user_proj)
        
        # Prediction
        combined = torch.cat([song_proj, cross_out], dim=1)
        output = self.predictor(combined)
        
        return output.squeeze()