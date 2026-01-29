import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# TRỤ CỘT 3: DATA AUGMENTATION (TĂNG CƯỜNG DỮ LIỆU) - MASTERCLASS
# -----------------------------------------------------------------------------
# SYNTX & LOGIC Masterclass:
# 1. np.clip: Cực kỳ quan trọng! Khi cộng/trừ pixel, giá trị có thể vượt quá 255 hoặc nhỏ hơn 0.
#    Cú pháp: np.clip(array, min, max) giúp giữ giá trị luôn trong biên [0, 255].
# 2. .astype(np.int16): Khi tính toán trung gian (như thêm nhiễu), ta dùng kiểu dữ liệu 
#    16-bit để tránh lỗi tràn số (overflow) trước khi đưa về 8-bit (uint8).
# 3. Random Seed: Giúp kết quả Augmentation có thể lặp lại được (dễ debug).
# =============================================================================

# Dùng Path để code chạy được trên mọi OS
IMG_PATH = Path("../../data/raw/img/000.jpg")

class MasterclassAugmenter:
    """
    Class chứa các bộ công cụ 'biến hóa' hình ảnh cho OCR.
    Mục tiêu: Dạy AI cách 'sống sót' trong thế giới thực đầy nhiễu.
    """
    def __init__(self, image):
        # Syntax: .copy() để đảm bảo không làm hỏng ảnh gốc bên ngoài
        self.image = image.copy()
        self.h, self.w = image.shape[:2]

    def add_brightness_contrast(self, brightness=30, contrast=1.2):
        """
        Lý thuyết: g(x) = alpha * f(x) + beta.
        - alpha (contrast): Phóng đại sự khác biệt giữa các pixel.
        - beta (brightness): Đẩy toàn bộ pixel lên một mức sáng mới.
        Syntax: cv2.convertScaleAbs là hàm tối ưu cho phép toán này.
        """
        # alpha > 1: Tăng tương phản | beta > 0: Tăng độ sáng
        return cv2.convertScaleAbs(self.image, alpha=contrast, beta=brightness)

    def add_gaussian_noise(self, sigma=25):
        """
        Lý thuyết: Mô phỏng nhiễu từ cảm biến máy ảnh (Digital Noise).
        Logic: Tạo một ma trận số ngẫu nhiên theo phân phối hình chuông (Gaussian) 
        rồi cộng trực tiếp vào ảnh.
        """
        # Syntax: np.random.normal(mình, độ_lệch, kích_thước)
        # Tại sao dùng int16? Vì pixel (255) + noise (30) = 285 (vượt uint8).
        noise = np.random.normal(0, sigma, self.image.shape).astype(np.int16)
        noisy_img = self.image.astype(np.int16) + noise
        
        # Syntax: Luôn phải clip và đưa về uint8 để OpenCV hiểu được
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    def add_motion_blur(self, kernel_size=15):
        """
        Lý thuyết: Mô phỏng việc máy ảnh bị rung theo một hướng nhất định.
        Logic: Tạo một Kernel đường thẳng (Identity matrix) xéo và trượt nó lên ảnh.
        """
        # Tạo kernel làm mờ chuyển động theo hướng nằm ngang
        kernel_h = np.zeros((kernel_size, kernel_size))
        kernel_h[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel_h /= kernel_size # Chuẩn hóa để độ sáng ảnh không thay đổi
        
        # cv2.filter2D: Hàm tổng quát để áp dụng bất kỳ ma trận Kernel nào lên ảnh
        return cv2.filter2D(self.image, -1, kernel_h)

    def add_perspective_tilt(self):
        """
        Lý thuyết: Mô phỏng việc chụp tài liệu bị xiên (không vuông góc).
        Logic: Chọn 4 điểm gốc và 'kéo' chúng sang 4 vị trí mới (Homography).
        """
        pts1 = np.float32([[50, 50], [self.w-50, 50], [50, self.h-50], [self.w-50, self.h-50]])
        # Bé bớt 1 góc để làm nó trông như bị nghiêng vào trong
        pts2 = np.float32([[0, 0], [self.w, 30], [0, self.h], [self.w, self.h-30]])
        
        # Tìm ma trận biến đổi phối cảnh 3x3
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(self.image, M, (self.w, self.h))

def run_day6_masterclass():
    # Đọc ảnh gốc
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        print("[ERROR] Không tìm thấy ảnh 000.jpg")
        return

    # Khởi tạo bộ augmenter
    aug = MasterclassAugmenter(img)

    # Thực hiện các phép biến thể
    var1 = aug.add_brightness_contrast(brightness=40, contrast=1.3)
    var2 = aug.add_gaussian_noise(sigma=30)
    var3 = aug.add_motion_blur(kernel_size=11)
    var4 = aug.add_perspective_tilt()

    # HIỂN THỊ SO SÁNH (Visual Debug)
    titles = ['Gốc', '1. Sáng & Tương phản', '2. Nhiễu Kỹ thuật số', '3. Rung tay (Motion Blur)', '4. Chụp Xiên (Perspective)']
    images = [img, var1, var2, var3, var4]

    plt.figure(figsize=(20, 12))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        # Chú thích Syntax: OpenCV dùng BGR, Matplotlib dùng RGB nên cần [:,:,::-1]
        plt.imshow(images[i][:,:,::-1])
        plt.title(titles[i], fontsize=14)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("[SUCCESS] Augmentation Masterclass hoàn tất! Hãy soi kỹ các nét chữ bị biến đổi.")

if __name__ == "__main__":
    run_day6_masterclass()
