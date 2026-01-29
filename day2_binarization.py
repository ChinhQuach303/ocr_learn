import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Cấu hình đường dẫn
IMG_PATH = Path("../../data/raw/img/000.jpg")

def test_binarization():
    if not IMG_PATH.exists():
        print(f"Lỗi: Không tìm thấy file {IMG_PATH}")
        return

    # 1. Đọc ảnh và chuyển sang Grayscale
    image = cv2.imread(str(IMG_PATH))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Global Thresholding (Ngưỡng cố định)
    # Nếu pixel > 127 thì thành 255 (trắng), ngược lại là 0 (đen)
    _, thresh_global = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 3. Otsu's Thresholding
    # Tự động tính toán ngưỡng tối ưu dựa trên histogram của ảnh
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Adaptive Thresholding (Gaussian)
    # Tính ngưỡng cho từng vùng nhỏ của ảnh (cực kỳ tốt cho ảnh bị đổ bóng)
    # ảnh đầu vào, cách tính ngưỡng, kiểu nhị phân, block_size: kích thước vùng lân cận và hằng số C
    thresh_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    # Hiển thị kết quả so sánh
    titles = ['Ảnh Gốc (Gray)', 'Global (127)', 'Otsu Theory', 'Adaptive Gaussian']
    images = [gray, thresh_global, thresh_otsu, thresh_adaptive]

    plt.figure(figsize=(18, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_binarization()
