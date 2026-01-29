import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMG_PATH = Path("../../data/raw/img/000.jpg")

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Đảo ngược màu để chữ là màu trắng (cần cho việc tìm góc)
    gray = cv2.bitwise_not(gray)
    
    # Tìm tất cả các pixel có màu > 0
    coords = np.column_stack(np.where(gray > 0))
    
    # Tìm hình chữ nhật bao quanh tối thiểu có xoay (minAreaRect)
    angle = cv2.minAreaRect(coords)[-1]
    
    # Xử lý góc của OpenCV
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    print(f"Góc nghiêng phát hiện: {angle:.2f} độ")
    
    # Xoay ảnh
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

if __name__ == "__main__":
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        print("Không đọc được ảnh")
    else:
        # Giả lập ảnh bị nghiêng 5 độ để test nếu ảnh gốc đã thẳng
        (h, w) = img.shape[:2]
        M_test = cv2.getRotationMatrix2D((w//2, h//2), 5, 1.0)
        skewed_img = cv2.warpAffine(img, M_test, (w, h))

        deskewed_img = deskew(skewed_img)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(skewed_img, cv2.COLOR_BGR2RGB)); plt.title("Ảnh bị nghiêng")
        plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(deskewed_img, cv2.COLOR_BGR2RGB)); plt.title("Ảnh đã chỉnh thẳng")
        plt.show()
