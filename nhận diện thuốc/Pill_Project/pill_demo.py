"""
PILL RECOGNITION DEMO - CG-IMIF MODEL
Chạy trên máy tính local với CustomTkinter
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import os
from pathlib import Path

# ============================================
# CẤU HÌNH GIAO DIỆN
# ============================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ============================================
# ĐỊNH NGHĨA MODEL CG-IMIF (GIỐNG CODE TRAINING)
# ============================================
class HistEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): 
        return self.net(x)


class CGIMIF_CPU(nn.Module):
    def __init__(self, num_classes, hist_bins=32, fusion_dim=256):
        super().__init__()
        import torchvision.models as models
        hist_input_dim = hist_bins * 3
        
        self.backbone = models.mobilenet_v3_small(weights=None)
        self.backbone.classifier = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.vis_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.hist_encoder = HistEncoder(hist_input_dim, output_dim=256)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, num_classes),
        )
    
    def forward(self, rgb, edge, hist):
        feat_rgb = self.backbone.features(rgb)
        feat_rgb = self.global_pool(feat_rgb)
        feat_rgb = self.vis_proj(feat_rgb)
        
        feat_edge = self.backbone.features(edge)
        feat_edge = self.global_pool(feat_edge)
        feat_edge = self.vis_proj(feat_edge)
        
        feat_vis = (feat_rgb + feat_edge) / 2
        feat_hist = self.hist_encoder(hist)
        feat = torch.cat([feat_vis, feat_hist], dim=1)
        return self.classifier(feat)


# ============================================
# HÀM TIỀN XỬ LÝ ẢNH
# ============================================
def compute_color_histogram(img_bgr, bins=32):
    hist_features = []
    for ch in range(3):
        hist = cv2.calcHist([img_bgr], [ch], None, [bins], [0, 256])
        hist_features.append(hist.flatten())
    hist_vec = np.concatenate(hist_features)
    hist_vec = hist_vec / (hist_vec.sum() + 1e-7)
    return hist_vec.astype(np.float32)

def compute_edge_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return cv2.merge([edges, edges, edges])

def preprocess_image(img_path, img_size=128):
    """
    Tiền xử lý ảnh với nhiều phương pháp đọc ảnh dự phòng
    """
    import os
    from pathlib import Path
    
    # 1. Kiểm tra file tồn tại
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {img_path}")
    
    # 2. Kiểm tra kích thước file
    file_size = img_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"File rỗng: {img_path}")
    
    # 3. Thử đọc ảnh với nhiều phương pháp
    img_bgr = None
    
    # Phương pháp 1: OpenCV đọc trực tiếp
    img_bgr = cv2.imread(str(img_path))
    
    # Phương pháp 2: Nếu OpenCV không đọc được, thử với PIL
    if img_bgr is None:
        try:
            from PIL import Image
            import numpy as np
            
            pil_img = Image.open(str(img_path))
            # Chuyển PIL sang numpy array (RGB)
            img_rgb = np.array(pil_img.convert('RGB'))
            # Chuyển RGB sang BGR cho OpenCV
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            print(f"✅ Đọc ảnh thành công bằng PIL")
        except Exception as e:
            print(f"❌ PIL cũng không đọc được: {e}")
    
    # Phương pháp 3: Đọc file binary và decode
    if img_bgr is None:
        try:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                print(f"✅ Đọc ảnh thành công bằng imdecode")
        except Exception as e:
            print(f"❌ imdecode cũng thất bại: {e}")
    
    # Nếu vẫn không đọc được, báo lỗi
    if img_bgr is None:
        raise ValueError(f"Không thể đọc ảnh bằng bất kỳ phương pháp nào: {img_path}")
    
    # Kiểm tra ảnh có hợp lệ không
    if img_bgr.shape[0] == 0 or img_bgr.shape[1] == 0:
        raise ValueError(f"Ảnh có kích thước không hợp lệ: {img_bgr.shape}")
    
    print(f"✅ Đã đọc ảnh thành công: {img_path.name} | Shape: {img_bgr.shape}")
    
    # Tiếp tục xử lý như code gốc
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Transform cho RGB
    transform_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Transform cho Edge
    transform_edge = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    rgb = transform_rgb(img_rgb).unsqueeze(0)
    edge = transform_edge(compute_edge_image(img_bgr)).unsqueeze(0)
    hist = torch.from_numpy(compute_color_histogram(img_bgr)).unsqueeze(0)
    
    return rgb, edge, hist


# ============================================
# LỚP ỨNG DỤNG DEMO
# ============================================
class PillDemoApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("NHẬN DIỆN VIÊN THUỐC - CG-IMIF")
        self.window.geometry("1000x750")
        self.window.resizable(False, False)
        
        # Load model và labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.labels = {}
        self.load_model_and_labels()
        
        # Thiết lập giao diện
        self.setup_ui()
        
    def load_model_and_labels(self):
        """Load model đã train và label mapping"""
        try:
            # Đường dẫn file
            model_path = "best_cgimif_cpu.pth"
            label_path = "label_mapping.json"
            
            if not os.path.exists(model_path):
                messagebox.showerror("Lỗi", f"Không tìm thấy file model: {model_path}\nVui lòng đặt file trong cùng thư mục")
                return
            
            if not os.path.exists(label_path):
                messagebox.showerror("Lỗi", f"Không tìm thấy file label: {label_path}")
                return
            
            # Load labels
            with open(label_path, 'r') as f:
                self.labels = json.load(f)
            
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            num_classes = checkpoint.get('num_classes', 108)
            
            self.model = CGIMIF_CPU(num_classes=num_classes, hist_bins=32, fusion_dim=256)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded | Device: {self.device}")
            print(f"✅ Classes: {num_classes}")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể load model:\n{str(e)}")
    
    def setup_ui(self):
        # Tiêu đề
        title = ctk.CTkLabel(
            self.window,
            text="NHẬN DIỆN VIÊN THUỐC",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.pack(pady=20)
        
        # Subtitle
        subtitle = ctk.CTkLabel(
            self.window,
            text="CG-IMIF Model | 108 loại thuốc | VAIPE-Pill Dataset",
            font=ctk.CTkFont(size=12)
        )
        subtitle.pack()
        
        # Frame chính
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # === BÊN TRÁI: KHU VỰC ẢNH ===
        left_frame = ctk.CTkFrame(main_frame, width=450, height=450)
        left_frame.pack(side="left", padx=20, pady=20, fill="both", expand=True)
        left_frame.pack_propagate(False)
        
        # Hiển thị ảnh
        self.image_label = ctk.CTkLabel(
            left_frame,
            text="📷 Chưa có ảnh\n\nClick 'Chọn ảnh' để tải lên",
            font=ctk.CTkFont(size=14),
            fg_color="#2b2b2b",
            corner_radius=10
        )
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # === BÊN PHẢI: KẾT QUẢ ===
        right_frame = ctk.CTkFrame(main_frame, width=400, height=450)
        right_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)
        
        # Tiêu đề kết quả
        result_title = ctk.CTkLabel(
            right_frame,
            text="KẾT QUẢ NHẬN DIỆN",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        result_title.pack(pady=10)
        
        # Kết quả chính
        self.result_label = ctk.CTkLabel(
            right_frame,
            text="--",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="green"
        )
        self.result_label.pack(pady=10)
        
        # Độ tin cậy
        self.confidence_label = ctk.CTkLabel(
            right_frame,
            text="Độ tin cậy: --",
            font=ctk.CTkFont(size=14)
        )
        self.confidence_label.pack()
        
        # Top 5 dự đoán
        top5_title = ctk.CTkLabel(
            right_frame,
            text="Top 5 dự đoán",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        top5_title.pack(pady=(20, 5))
        
        self.top5_frame = ctk.CTkFrame(right_frame)
        self.top5_frame.pack(fill="x", padx=10, pady=5)
        
        self.top5_labels = []
        for i in range(5):
            lbl = ctk.CTkLabel(
                self.top5_frame,
                text=f"{i+1}. ---",
                font=ctk.CTkFont(size=12),
                anchor="w"
            )
            lbl.pack(fill="x", pady=2)
            self.top5_labels.append(lbl)
        
        # === NÚT BẤM ===
        button_frame = ctk.CTkFrame(self.window)
        button_frame.pack(pady=20)
        
        self.btn_open = ctk.CTkButton(
            button_frame,
            text="📁 CHỌN ẢNH",
            command=self.open_image,
            width=150,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.btn_open.pack(side="left", padx=10)
        
        self.btn_predict = ctk.CTkButton(
            button_frame,
            text="🔍 NHẬN DIỆN",
            command=self.predict,
            width=150,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled",
            fg_color="#2b5e2b"
        )
        self.btn_predict.pack(side="left", padx=10)
        
        # Thanh trạng thái
        self.status_label = ctk.CTkLabel(
            self.window,
            text="✅ Sẵn sàng. Vui lòng chọn ảnh viên thuốc.",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=10)
        
        self.current_image_path = None
        
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh viên thuốc",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.btn_predict.configure(state="normal")
            self.status_label.configure(text=f"✅ Đã tải ảnh: {os.path.basename(file_path)}")
            self.result_label.configure(text="--")
            self.confidence_label.configure(text="Độ tin cậy: --")
            for lbl in self.top5_labels:
                lbl.configure(text=f"{self.top5_labels.index(lbl)+1}. ---")
    
    def display_image(self, path):
        try:
            img = Image.open(path)
            # Resize để vừa khung
            img.thumbnail((430, 430))
            
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            self.status_label.configure(text=f"❌ Lỗi hiển thị ảnh: {str(e)}")
    
    def predict(self):
        if self.current_image_path is None:
            return
        
        if self.model is None:
            messagebox.showerror("Lỗi", "Model chưa được tải")
            return
        
        self.status_label.configure(text="⏳ Đang nhận diện...")
        self.window.update()
        
        try:
            # Tiền xử lý ảnh
            rgb, edge, hist = preprocess_image(self.current_image_path)
            rgb = rgb.to(self.device)
            edge = edge.to(self.device)
            hist = hist.to(self.device)
            
            # Dự đoán
            with torch.no_grad():
                outputs = self.model(rgb, edge, hist)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top5_probs, top5_indices = torch.topk(probs, 5)
            
            # Lấy kết quả
            top5_probs = top5_probs[0].cpu().numpy()
            top5_indices = top5_indices[0].cpu().numpy()
            
            # Class dự đoán chính
            pred_idx = top5_indices[0]
            pred_prob = top5_probs[0]
            
            # Lấy tên class từ label mapping
            # Lưu ý: label_mapping.json có cấu trúc {class_id: class_name}
            class_name = self.labels.get(str(pred_idx), f"Class_{pred_idx}")
            
            # Hiển thị kết quả
            self.result_label.configure(text=f"{class_name}", text_color="green")
            self.confidence_label.configure(text=f"Độ tin cậy: {pred_prob:.2%}")
            
            # Hiển thị Top 5
            for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
                name = self.labels.get(str(idx), f"Class_{idx}")
                self.top5_labels[i].configure(
                    text=f"{i+1}. {name} ({prob:.1%})",
                    text_color="white" if i == 0 else "gray"
                )
            
            self.status_label.configure(
                text=f"✅ Nhận diện thành công! Kết quả: {class_name} (độ tin cậy {pred_prob:.1%})"
            )
            
        except Exception as e:
            self.status_label.configure(text=f"❌ Lỗi nhận diện: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể nhận diện ảnh:\n{str(e)}")
    
    def run(self):
        self.window.mainloop()


# ============================================
# KHỞI CHẠY ỨNG DỤNG
# ============================================
if __name__ == "__main__":
    app = PillDemoApp()
    app.run()