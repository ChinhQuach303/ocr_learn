import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# TRỤ CỘT 1 & 3: IMAGE REPRESENTATION & SIGNAL PROCESSING
# LÝ THUYẾT:
# Tại sao cần Nhị phân hóa (Binarization)?
# AI không cần màu sắc của giấy, nó chỉ cần sự tương phản giữa "Chữ" và "Nền".
# 1. Global Threshold: Dùng 1 ngưỡng duy nhất (VD: 127). Yếu khi ảnh bị bóng đổ.
# 2. Otsu: Tự động tính ngưỡng từ Histogram. Tốt cho ảnh có nền sạch.
# 3. Adaptive: Tính ngưỡng cho từng vùng nhỏ. "Cứu cánh" cho ảnh chụp thực tế.
# =============================================================================

IMG_PATH = Path("../../data/raw/img/001.jpg")

def masterclass_binarization(img_path):
    # Đọc ảnh gốc
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Không thấy ảnh tại {img_path}")
        
    # --- PHẦN 1: TIỀN XỬ LÝ (PRE-PROCESSING) ---
    # Chuyển sang Grayscale (Hệ màu 1 kênh - Gray)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur: Làm mờ nhẹ để giảm nhiễu hạt trước khi nhị phân.
    # Kernel (5,5). Tại sao? Để các cạnh chữ mượt hơn, tránh bị răng cưa khi tách ngưỡng.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- PHẦN 2: CÁC KỸ THUẬT NHỊ PHÂN (THE CORE) ---
    
    # 1. Global Thresholding (Cố định tại 127)
    _, thresh_global = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    # 2. Otsu's Thresholding (AI tự tìm ngưỡng)
    # Nó quét toàn bộ Histogram để tìm điểm chia cắt tốt nhất giữa 2 cụm pixel.
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Adaptive Thresholding (Gaussian Weighted)
    # Block Size=11: Kích thước vùng lân cận để tính ngưỡng.
    # C=2: Hằng số trừ đi từ trung bình (hiệu chỉnh độ sáng).
    thresh_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    # --- PHẦN 3: KỸ THUẬT NÂNG CAO (BLACKHAT) ---
    # Blackhat = Closing - Original. 
    # Nó giúp trích xuất các nét tối (chữ) trên nền sáng, đặc biệt hiệu quả khi bóng đổ nặng.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
    enhanced_bin = cv2.adaptiveThreshold(
        cv2.add(blurred, blackhat), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    # --- PHẦN 4: VISUAL COMPARISON ---
    titles = ['Gốc (Gray)', 'Global (127)', 'Otsu (Auto)', 'Adaptive (Pro)', 'Adaptive + Blackhat']
    images = [gray, thresh_global, thresh_otsu, thresh_adaptive, enhanced_bin]

    plt.figure(figsize=(20, 8))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
        
    plt.tight_layout()
    print("[HƯỚNG DẪN]: Hãy để ý Adaptive + Blackhat giữ được nét chữ tốt nhất ở vùng bị bóng tối!")
    plt.show()

if __name__ == "__main__":
    masterclass_binarization(IMG_PATH)
