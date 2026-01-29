import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMG_PATH = Path("../../data/raw/img/000.jpg")

def augment_image(image):
    # 1. Thay đổi độ sáng (Brightness)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 50) # Tăng độ sáng
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 2. Làm mờ (Blur)
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    # 3. Thêm nhiễu Gauss (Gaussian Noise)
    noise = np.random.normal(0, 25, image.shape).astype(np.int16)
    noisy = cv2.add(image.astype(np.int16), noise)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return bright, blurred, noisy

if __name__ == "__main__":
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        print("Lỗi đọc ảnh")
    else:
        bright, blurred, noisy = augment_image(img)
        
        titles = ['Gốc', 'Độ sáng +', 'Làm mờ', 'Nhiễu Gauss']
        imgs = [img, bright, blurred, noisy]
        
        plt.figure(figsize=(16, 4))
        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.axis('off')
        plt.show()
