import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMG_PATH = Path("../../data/raw/img/000.jpg")

def test_denoising():
    image = cv2.imread(str(IMG_PATH))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Tạo ảnh nhị phân để demo Morphological
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 1. Median Blur (Khử nhiễu muối tiêu rất tốt)
    denoised = cv2.medianBlur(binary, 3)

    # 2. Erosion (Làm gầy nét chữ - loại bỏ các đốm trắng nhỏ)
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)

    # 3. Dilation (Làm dày nét chữ - lấp đầy các lỗ hổng trong chữ)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # 4. Opening (Erosion sau đó Dilation - dùng để xóa nhiễu nền)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Hiển thị
    titles = ['Nhị phân gốc', 'Median Blur', 'Erosion', 'Dilation', 'Opening']
    images = [binary, denoised, eroded, dilated, opening]

    plt.figure(figsize=(20, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    test_denoising()
