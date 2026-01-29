import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_sauvola

# =============================================================================
# TRỤ CỘT 3: SIGNAL PROCESSING & MORPHOLOGY
# LÝ THUYẾT:
# Sau khi nhị phân, ảnh thường bị "vụn" (nhiễu muối tiêu) hoặc "đứt nét".
# 1. Median Blur: Cực tốt để xóa các đốm nhiễu li ti (nhiễu hạt).
# 2. Morphological Operations: 
#    - Erosion (Co): Xóa nhiễu nền nhưng làm gầy chữ.
#    - Dilation (Dãn): Lấp đầy lỗ hổng trong chữ nhưng làm dày chữ.
#    - Opening: Erosion rồi Dilation (Xóa nhiễu mà giữ Form chữ).
#    - Closing: Dilation rồi Erosion (Nối nét đứt).
# =============================================================================

IMG_PATH = Path("../../data/raw/img/005.jpg")

def masterclass_denoising(img_path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Dùng Otsu để làm ảnh nhị phân nền
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Median Filter (Số lẻ: 3, 5, 7...)
    # Nó lấy giá trị trung vị trong vùng cửa sổ. Giúp triệt tiêu cực tốt các điểm trắng cô độc.
    median = cv2.medianBlur(binary, 3)

    # 3. Morphological - Opening (Xóa nhiễu li ti trên nền)
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 4. Morphological - Closing (Nối nét chữ bị đứt)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 5. Sauvola (Thuật toán đỉnh cao từ skimage)
    # Tốt hơn cả Adaptive trong việc giữ nét chữ thanh mảnh của tài liệu cổ.
    thresh_sauvola = threshold_sauvola(gray, window_size=25, k=0.2)
    binary_sauvola = (gray < thresh_sauvola).astype(np.uint8) * 255

    # --- PHẦN 4: VISUAL DEBUG ---
    images = [binary, median, opening, closing, binary_sauvola]
    titles = ['Gốc (Nhị phân)', 'Median Blur', 'Opening (Clean)', 'Closing (Connect)', 'Sauvola (Top Tier)']

    plt.figure(figsize=(20, 8))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    print("[HƯỚNG DẪN]: So sánh Opening và Sauvola để thấy cách chúng xử lý nhiễu khác nhau!")
    plt.show()

if __name__ == "__main__":
    masterclass_denoising(IMG_PATH)
