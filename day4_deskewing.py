import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# TRỤ CỘT 2: HÌNH HỌC & PHÉP BIẾN ĐỔI (HOUGH LINE TRANSFORM)
# LÝ THUYẾT:
# Tại sao dùng Hough Line Transform để Deskew?
# Thay vì nhìn cả khối ảnh, chúng ta đi tìm các "đoạn thẳng" (line segments).
# Trong văn bản, các dòng chữ chính là các tập hợp điểm tạo thành đoạn thẳng nằm ngang.
# 
# LƯU Ý VỀ LỖI BẠN GẶP:
# Lỗi 'invalid value encountered in scalar divide' xuất hiện khi:
# 1. Threshold của HoughLines quá cao -> Không tìm thấy đường nào -> List angles trống.
# 2. np.median([]) gây lỗi vì không có gì để tính toán. 
# CHẤT LƯỢNG CODE: Luôn luôn kiểm tra dữ liệu trước khi tính toán thống kê!
# =============================================================================

IMG_PATH = Path("../../data/raw/img/000.jpg")

def masterclass_hough_deskew(image_path):
    # Đọc ảnh
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Không tìm thấy ảnh tại: {image_path}")
        return
        
    # --- PHẦN 1: TIỀN XỬ LÝ ĐỂ TÌM CẠNH ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Canny Edge Detection: Giúp làm nổi bật các cạnh của chữ.
    # Các hàm Hough hoạt động tốt nhất trên ảnh cạnh (Edges).
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # --- PHẦN 2: HOUGH LINES (PHÁT HIỆN ĐƯỜNG THẲNG) ---
    # cv2.HoughLinesP (Probabilistic Hough Transform) hiệu quả hơn HoughLines thường.
    # threshold=100: Số điểm tối thiểu để công nhận 1 đường thẳng.
    # minLineLength=100: Độ dài tối thiểu của đoạn thẳng.
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    angles = []
    
    # --- PHẦN 3: XỬ LÝ LOGIC VÀ PHÒNG LỖI (FIX WARNING) ---
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Tính góc của đoạn thẳng dùng arctan2(dy, dx)
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            
            # OCR thường chỉ nghiêng ít (khoảng -45 đến 45 độ)
            if -45 < angle < 45:
                angles.append(angle)
    
    # KIỂM TRA QUAN TRỌNG: Nếu không tìm thấy đường thẳng nào, mặc định là 0 độ.
    if len(angles) > 0:
        skew_angle = np.median(angles)
        print(f"[LOG] Tìm thấy {len(angles)} đoạn thẳng. Góc trung vị: {skew_angle:.2f} độ")
    else:
        skew_angle = 0.0
        print("[WARNING] Không tìm thấy đoạn thẳng nào thỏa mãn! Giữ nguyên góc 0.0")

    # --- PHẦN 4: XOAY ẢNH (AFFINE TRANSFORMATION) ---
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated, edges, skew_angle

if __name__ == "__main__":
    # Chạy thử
    result, debug_edges, angle = masterclass_hough_deskew(IMG_PATH)
    
    # Hiển thị kết quả
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(str(IMG_PATH))[:,:,::-1])
    plt.title("1. Ảnh gốc")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(debug_edges, cmap='gray')
    plt.title("2. Canny Edges (Đầu vào Hough)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(result[:,:,::-1])
    plt.title(f"3. Kết quả (Xoay {angle:.2f} độ)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()