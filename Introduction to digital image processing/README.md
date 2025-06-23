
# Nhập Môn Xử Lý Ảnh Số - Lab 2  
## **Biến Đổi Cường Độ Ảnh Grayscale & Phân Tích Bit-Plane**
**Sinh viên thực hiện:** Example
**MSSV:** Example  
**Môn học:** Nhập môn Xử lý ảnh số  
**Giảng viên:** Đỗ Hữu Quân


## Giới thiệu

Bài lab này nhằm mục đích thực hiện **các phép biến đổi cường độ trên ảnh xám** – là những thao tác cơ bản, nền tảng trong lĩnh vực xử lý ảnh số. Các biến đổi này giúp:
- Tăng cường độ tương phản
- Làm nổi bật các chi tiết khó quan sát
- Trích xuất đặc trưng ảnh 


## Công nghệ sử dụng


- **Python**: Ngôn ngữ chính                           
- **Pillow (PIL)**: Đọc, chuyển đổi, và lưu ảnh              
- **NumPy**: Xử lý ảnh dưới dạng mảng số học          
- **ImageIO**: Đọc file ảnh với định dạng hiện đại      
- **Matplotlib**: Hiển thị ảnh trực quan                   


## Chi tiết các phép biến đổi & công thức

### 1. Biến đổi ảnh đảo ngược (Negative/Inverse Transformation)

**Mục đích:**  
- Đảo ngược giá trị pixel, biến vùng sáng thành tối và ngược lại
- Làm nổi bật các chi tiết mờ trong vùng tối, tăng khả năng nhận diện vật thể (thường dùng cho ảnh X-ray)

**Công thức toán học:**  
```math
s = L - 1 - r
```

- `r`: Giá trị pixel gốc (0–255 với ảnh 8 bit)
- `L`: Số mức xám (thường là 256)
- `s`: Giá trị pixel mới

**Ví dụ:**  
- Nếu pixel gốc có giá trị 100 → giá trị mới = 255 - 100 = 155

**Code chính:**  
```python
img_np = np.array(img)
negative_img = 255 - img_np
```
---

### 2. Biến đổi logarit (Log Transformation)

**Mục đích:**  
- Khuếch đại các giá trị pixel nhỏ (làm rõ chi tiết trong vùng tối)
- Giảm thiểu tác động các giá trị pixel lớn (nền sáng không bị lóa)
- Ứng dụng trong ảnh vệ tinh, ảnh y khoa...

**Công thức toán học:**  
```math
s = c * log(1 + r)
```
- `c`: Hệ số chuẩn hóa (`c = 255 / log(1 + max(r))`)
- `r`: Giá trị pixel gốc (0–255)
- `s`: Giá trị pixel mới

**Ví dụ:**  
- Nếu `r` nhỏ, log(1+r) thay đổi nhanh, vùng tối sẽ được làm sáng rõ rệt.

**Code chính:**  
```python
c = 255 / np.log(1 + np.max(img_np))
log_img = c * np.log(1 + img_np)
log_img = np.array(log_img, dtype=np.uint8)
```
---

### 3. Biến đổi hàm mũ/Gamma (Power-law/Gamma Transformation)

**Mục đích:**  
- Hiệu chỉnh gamma cho màn hình hiển thị, tăng hoặc giảm độ sáng tổng thể của ảnh
- Có thể làm sáng (gamma < 1) hoặc làm tối (gamma > 1) tùy mục đích
- Ứng dụng nhiều trong truyền hình, xử lý ảnh số

**Công thức toán học:**  
```math
s = c * r^γ
```
- `γ`: Hệ số gamma
- `c`: Hệ số chuẩn hóa (thường là 1, hoặc 255/ (255^γ) để đưa kết quả về 0-255)
- `r`: Giá trị pixel gốc (chuẩn hóa về 0-1 nếu cần)

**Lưu ý:**  
- Nếu γ < 1: Ảnh sáng lên  
- Nếu γ > 1: Ảnh tối đi

**Ví dụ:**  
- Với gamma = 0.5, pixel gốc 100: s = 255 * (100/255)^0.5 ≈ 159

**Code chính:**  
```python
gamma = 0.5  # hoặc 2.0
img_norm = img_np / 255.0
gamma_img = np.power(img_norm, gamma)
gamma_img = np.uint8(gamma_img * 255)
```
---

### 4. Phân tích mặt phẳng bit (Bit-Plane Slicing)

**Mục đích:**  
- Phân tích ảnh theo từng bit trong mỗi pixel
- Cho biết mức độ đóng góp của từng bit vào tổng thể độ sáng
- Dùng để phát hiện watermark, các đặc trưng ẩn trong ảnh (steganography) hoặc làm rõ chi tiết cạnh, noise...

**Nguyên lý:**  
- Ảnh 8-bit: mỗi pixel có 8 bit (bit 0 là LSB, bit 7 là MSB)
- Mỗi mặt phẳng bit là một ảnh nhị phân, pixel giá trị 0 hoặc 1 tương ứng với bit ở vị trí đó

**Code chính:**  
```python
bit_plane = 3  # từ 0 đến 7
bp_img = (img_np >> bit_plane) & 1
bp_img = bp_img * 255
```

**Ý nghĩa từng bit:**  
- Bit cao (6, 7): Quyết định hình dạng, chi tiết chính của ảnh
- Bit thấp (0, 1): Chủ yếu là nhiễu hoặc chi tiết nhỏ

---

## Cấu trúc file

```
├── main.ipynb      
├── image.png        
├── README.md       
```

## Hướng dẫn

### 1. Cài đặt môi trường

Cài Python, sau đó cài các thư viện:

```bash
pip install pillow numpy matplotlib imageio
```

### 2. Chạy notebook

- Mở Jupyter Notebook trên VSCode/Colab
- Chạy từng cell để xem kết quả của các thuật toán
- Nếu có lỗi không tìm thấy ảnh, đảm bảo file ảnh đã được đặt đúng vị trí

### 3. Tùy chỉnh tham số

- Thay đổi giá trị `gamma, c, bit-plane` trong từng cell để quan sát tác động thực tế

---

## Tài liệu tham khảo

- [Digital Image Processing - Rafael C. Gonzalez](https://www.amazon.com/Digital-Image-Processing-Rafael-Gonzalez/dp/013168728X)
- [Scikit-image: Image processing in Python](https://scikit-image.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- Slide bài giảng Nhập môn Xử lý ảnh số - Văn Lang University

---
