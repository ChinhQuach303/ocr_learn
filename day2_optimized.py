import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Cấu hình đường dẫn
IMG_PATH = Path("../../data/raw/img/001.jpg")
def modern_threshold_v2(img_path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pre-processing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Blackhat để lấy nét chữ từ vùng tối
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
    
    # Kết hợp để làm nổi bật chữ
    enhanced = cv2.add(blurred, blackhat)
    
    # Adaptive Threshold
    # Lưu ý: C=8 là khá lớn, nếu mất nét chữ bạn hãy giảm xuống 2 hoặc 3
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 3, 2
    )
    
    return gray, enhanced, thresh

if __name__ == "__main__":
    original, enh, result = modern_threshold_v2(IMG_PATH)
    
    # Hiển thị so sánh 3 bước
    titles = ['Original Gray', 'Enhanced (Blackhat)', 'Final Result']
    images = [original, enh, result]
    
    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()