import cv2
import numpy as np
from pathlib import Path

class OCRPreprocessor:
    def __init__(self, target_width=None):
        self.target_width = target_width

    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def binarize(self, gray):
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

    def denoise(self, binary):
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def process(self, image):
        # Đường ống xử lý (Pipeline)
        gray = self.grayscale(image)
        binary = self.binarize(gray)
        clean = self.denoise(binary)
        return clean

if __name__ == "__main__":
    # Test pipeline
    processor = OCRPreprocessor()
    img_path = Path("../../data/raw/img/000.jpg")
    img = cv2.imread(str(img_path))
    
    if img is not None:
        processed_img = processor.process(img)
        cv2.imshow("Original", cv2.resize(img, (600, 800)))
        cv2.imshow("Processed", cv2.resize(processed_img, (600, 800)))
        print("Đang hiển thị kết quả. Nhấn phím bất kỳ để thoát.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không tìm thấy ảnh")
