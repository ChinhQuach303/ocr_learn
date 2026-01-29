import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# TRỤ CỘT 7: INTEGRATED PREPROCESSING PIPELINE (KẾT THÚC TUẦN 1)
# -----------------------------------------------------------------------------
# SYNTX & LOGIC Masterclass:
# 1. Modular Design: Đóng gói các kỹ thuật vào các phương thức (methods) riêng biệt.
# 2. Pipeline Pattern: Dữ liệu chảy từ bước này sang bước kia (Raw -> Gray -> Deskew -> Binary).
# 3. Parameter Tuning: Khả năng tùy chỉnh tham số (Threshold, Kernel) cho từng loại ảnh.
# 4. Dictionary Mapping: Lưu trữ kết quả theo bộ (Original, Processed, ROIs).
# =============================================================================

# Đường dẫn dữ liệu
DATA_DIR = Path("../../data/raw")
IMG_PATH = DATA_DIR / "img" / "000.jpg"
BOX_PATH = DATA_DIR / "box" / "000.csv"

class ModernOCRPipeline:
    """
    Hệ thống Tiền xử lý OCR toàn diện - Kết tinh kiến thức Tuần 1.
    """
    def __init__(self):
        print("[INIT] Khởi tạo OCR Pipeline v1.0...")

    # --- BƯỚC 1: LÀM THẲNG (DESKEW - DAY 4) ---
    def deskew_image(self, image):
        """Logic: Dùng Hough Transform để tìm góc nghiêng và nắn thẳng."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            angles = [np.rad2deg(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0])) for l in lines]
            angle = np.median(angles)
            # Giới hạn góc để tránh xoay quá đà
            if abs(angle) < 45:
                (h, w) = image.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return image

    # --- BƯỚC 2: LÀM SẠCH (BINARIZE & DENOISE - DAY 2 & 3) ---
    def clean_image(self, image):
        """Logic: Chuyển xám -> Nhị phân Adaptive -> Khử nhiễu Morphology."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Nhị phân hóa thích nghi: Giải quyết vấn đề ánh sáng không đều
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Khử nhiễu Morphology: Xóa các đốm 'muối tiêu'
        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return clean

    # --- BƯỚC 3: PHÂN TÁCH VÙNG (ROI EXTRACTION - DAY 5) ---
    def get_rois_manual(self, image, box_path):
        """Logic: Cắt ảnh dựa trên tọa độ CSV có sẵn."""
        rois = []
        try:
            # Syntax: Đọc CSV bỏ qua lỗi định dạng dòng
            df = pd.read_csv(box_path, header=None, on_bad_lines='skip')
            for _, row in df.iterrows():
                pts = row.values[:8].reshape((-1, 2)).astype(int)
                x, y = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                # Syntax: Slicing [Y, X]
                roi = image[max(0, y):y2, max(0, x):x2]
                if roi.size > 0:
                    rois.append(roi)
        except Exception as e:
            print(f"[ERROR] Lỗi đọc box: {e}")
        return rois

    # --- HÀM THỰC THI CHÍNH (THE FLOW) ---
    def run(self, img_path, box_path):
        # 1. Load ảnh
        raw_img = cv2.imread(str(img_path))
        if raw_img is None: return None
        
        print(f"[PROCESS] Đang xử lý ảnh: {img_path.name}")
        
        # 2. Xoay thẳng ảnh
        straight_img = self.deskew_image(raw_img)
        
        # 3. Làm sạch ảnh (để nhận diện toàn cục hoặc debug)
        clean_img = self.clean_image(straight_img)
        
        # 4. Trích xuất các mẩu chữ (ROIs)
        rois = self.get_rois_manual(straight_img, box_path)
        
        return {
            "original": raw_img,
            "processed": clean_img,
            "rois": rois[:5] # Lấy 5 mẫu đầu để demo
        }

def show_final_results(res):
    if res is None: return
    
    plt.figure(figsize=(15, 10))
    
    # Hiển thị ảnh gốc
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(res["original"], cv2.COLOR_BGR2RGB))
    plt.title("1. Original Image (Raw)", fontsize=12)
    plt.axis('off')

    # Hiển thị ảnh đã xử lý (Deskew + Clean)
    plt.subplot(2, 2, 2)
    plt.imshow(res["processed"], cmap='gray')
    plt.title("2. Full Processed (Deskew + Binarize)", fontsize=12)
    plt.axis('off')

    # Hiển thị các ROI mẫu
    for i, roi in enumerate(res["rois"]):
        if i >= 4: break # Hiển thị tối đa 4 ROI ở hàng dưới
        plt.subplot(2, 4, 5 + i)
        if len(roi.shape) == 3:
            plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(roi, cmap='gray')
        plt.title(f"ROI Sample {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle("WEEK 1 FINAL PIPELINE: FROM RAW TO CLEAN DATA", fontsize=16)
    plt.show()

if __name__ == "__main__":
    # Khởi động Robot Pipeline
    pipeline = ModernOCRPipeline()
    results = pipeline.run(IMG_PATH, BOX_PATH)
    
    if results:
        print(f"[SUCCESS] Trích xuất được {len(results['rois'])} mẫu ROIs.")
        show_final_results(results)
