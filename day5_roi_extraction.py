import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# TRỤ CỘT 1 & 2: ROI EXTRACTION & AUTOMATIC DETECTION (CCL)
# -----------------------------------------------------------------------------
# SYNTX & LOGIC MASTERCLASS:
# 1. Pathlib (Path): Dùng '/' để nối đường dẫn, giúp code chạy được trên cả Windows/Linux.
#    Cú pháp: Path("folder") / "subfolder" / "file.txt"
# 2. Numpy Reshape: .reshape((-1, 2)) -> "-1" nghĩa là "tự tính số hàng dựa trên số cột là 2".
# 3. List Comprehension: [float(p) for p in parts] -> Cách tạo list nhanh và gọn trong Python.
# 4. Dictionary/JSON: Dùng để lưu trữ Metadata (nhãn văn bản).
# =============================================================================

# Chú thích Syntax: Dùng 'Path' giúp tránh lỗi dấu '\' trong Windows
DATA_RAW_DIR = Path("../../data/raw")
SAVE_DIR = Path("../../data/processed/week1_crops")

def denoise_roi(roi):
    """
    Sử dụng Morphology (Hình thái học) để làm sạch nhiễu.
    Syntax: np.ones((2,2), np.uint8) tạo ma trận toàn số 1, kiểu dữ liệu 8-bit không dấu.
    """
    kernel = np.ones((2, 2), np.uint8)
    # cv2.MORPH_OPEN = Erosion (mòn) sau đó Dilation (nở). 
    # Giúp triệt tiêu các chấm trắng nhỏ (nhiễu) mà không làm đổi kích thước vật thể chính.
    return cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

def binarize_roi(roi):
    """
    Chuyển ROI về dạng trắng đen sắc nét.
    Syntax: len(roi.shape) == 3 nghĩa là ảnh có 3 kênh màu (BGR).
    """
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # adaptiveThreshold: Tính ngưỡng cho từng vùng (block) 11x11 pixel.
    # C=2: Hằng số trừ đi từ trung bình để loại bỏ nhiễu nền xám.
    return cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

def process_roi(roi):
    """Pipeline làm rõ ROI: Nhị phân hóa -> Khử nhiễu."""
    binary = binarize_roi(roi)
    clean = denoise_roi(binary)
    return clean

def masterclass_roi_extraction(sample_id="000"):
    """
    Trích xuất vùng quan tâm dựa trên file Bounding Box (CSV).
    """
    img_path = DATA_RAW_DIR / "img" / f"{sample_id}.jpg"
    box_path = DATA_RAW_DIR / "box" / f"{sample_id}.csv"
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    image = cv2.imread(str(img_path))
    if image is None: return None

    # Syntax: 'with open' đảm bảo file được đóng ngay sau khi đọc xong (tránh rò rỉ bộ nhớ)
    data = []
    with open(box_path, 'r', encoding='utf-8') as f:
        for line in f:
            # .strip() xóa khoảng trắng thừa, .split(',') chia chuỗi thành danh sách
            parts = line.strip().split(',')
            if len(parts) >= 8:
                # Ép kiểu dữ liệu sang float để tính toán tọa độ chính xác
                coords = [float(p) for p in parts[:8]]
                data.append(coords)
    
    # Chuyển list thành DataFrame để dễ dàng thao tác theo hàng/cột
    boxes = pd.DataFrame(data)
    first_roi = None

    for i, row in boxes.iterrows():
        # Chú thích Logic: .min(axis=0) tìm giá trị nhỏ nhất theo chiều dọc (cột)
        pts = row.values[:8].reshape((-1, 2))
        x_min, y_min = pts.min(axis=0).astype(int)
        x_max, y_max = pts.max(axis=0).astype(int)

        # Syntax: image[y:y, x:x] -> Nhớ quy tắc: Y (Hàng) trước, X (Cột) sau
        roi = image[max(0, y_min):y_max, max(0, x_min):x_max]
        
        if roi.size > 0:
            if first_roi is None: first_roi = roi
            # i:03d -> Định dạng số ví dụ 001, 002... giúp sắp xếp file đẹp hơn
            cv2.imwrite(str(SAVE_DIR / f"{sample_id}_crop_{i:03d}.jpg"), roi)

    return first_roi

def roi_by_ccl(image_path):
    """
    TỰ ĐỘNG tìm vùng văn bản bằng Connected Component Labeling.
    Logic: Gom các pixel liên thông thành các 'nhãn' (labels).
    """
    img = cv2.imread(str(image_path))
    if img is None: return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # THRESH_BINARY_INV: Vì CCL tìm 'vật thể trắng' trên 'nền đen'
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Syntax: cv2.connectedComponentsWithStats trả về 4 giá trị quan trọng bao gồm
    # num_labels: Số lượng 'nhãn' (labels) tìm thấy
    # labels: Ma trận chứa thông tin về 'nhãn' của mỗi pixel
    # stats: Ma trận chứa thông số của mỗi 'nhãn'
    # centroids: Ma trận chứa tọa độ trung tâm của mỗi 'nhãn'
    # connectivity=8: Xét các pixel chạm nhau cả ở góc chéo.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    output = img.copy()
    count = 0
    # Loop từ 1 vì 0 luôn là nền (background)
    for i in range(1, num_labels):
        # Lấy thông số từ ma trận 'stats'
        # stats[i, cv2.CC_STAT_LEFT]: Tọa độ x của góc trái trên
        # stats[i, cv2.CC_STAT_TOP]: Tọa độ y của góc trái trên
        # stats[i, cv2.CC_STAT_WIDTH]: Chiều rộng
        # stats[i, cv2.CC_STAT_HEIGHT]: Chiều cao
        # stats[i, cv2.CC_STAT_AREA]: Kích thước (diện tích)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Logic Lọc (Filtering): 
        # - Area > 10: Bỏ các đốm nhiễu li ti.
        # - h < 200: Bỏ các vệt bẩn dài hoặc khung viền trang giấy.
        if 10 < area < 5000 and 5 < h < 200:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
            count += 1
            
    return output

def recursive_xy_cut(thresh, boxes):
    """
    Thuật toán Recursive XY Cut (Recursive X-Y Cut).
    Lý thuyết: Đây là thuật toán "chia để trị" dựa trên lược đồ hình chiếu (Projection Profile).
    1. Hình chiếu ngang: Tìm các khoảng trống (gap) giữa các dòng chữ.
    2. Hình chiếu dọc: Từ các dòng, tìm khoảng trống giữa các từ/cột.
    Syntax: np.sum(axis=1) tính tổng pixel theo hàng ngang (dùng cho X-cut).
    """
    # 1. Hình chiếu ngang (Horizontal Projection) - Tìm dòng
    # h_proj là 1 mảng 1 chiều chứa tổng số pixel của mỗi hàng
    h_proj = np.sum(thresh, axis=1)
    
    # Tìm các chỉ số hàng có chứa pixel chữ (tổng > 0)
    # Syntax: np.where trả về tuple, ta lấy phần tử [0]
    # upper_lines là 1 mảng 1 chiều chứa các chỉ số hàng có chứa pixel chữ
    upper_lines = np.where(h_proj > 0)[0]
    if len(upper_lines) == 0: return
    
    # Logic tách dòng: Tìm các đoạn liên tiếp trong upper_lines
    start_y = upper_lines[0]
    for i in range(1, len(upper_lines)):
        # Nếu khoảng cách giữa 2 pixel chữ > 5px, coi như là khoảng trống giữa 2 dòng
        # cái này nên tối ưu kiểu gì? thực tế thì phải tự động và tối ưu tốt hơn
        if upper_lines[i] - upper_lines[i-1] > 5:
            end_y = upper_lines[i-1]
            
            # 2. Xử lý từng dòng vừa tìm được để tìm hộp văn bản (Vertical Projection)
            # line_roi là 1 mảng 2 chiều chứa tổng số pixel của mỗi hàng trong dòng
            line_roi = thresh[start_y:end_y, :]
            # v_proj là 1 mảng 1 chiều chứa tổng số pixel của mỗi cột trong dòng
            v_proj = np.sum(line_roi, axis=0)
            # left_lines là 1 mảng 1 chiều chứa các chỉ số cột có chứa pixel chữ
            left_lines = np.where(v_proj > 0)[0]
            
            if len(left_lines) > 0:
                start_x = left_lines[0]
                for j in range(1, len(left_lines)):
                    # Nếu khoảng cách ngang > 10px, coi như là khoảng cách giữa các từ/cột
                    if left_lines[j] - left_lines[j-1] > 10:
                        end_x = left_lines[j-1]
                        boxes.append((start_x, start_y, end_x - start_x, end_y - start_y))
                        start_x = left_lines[j]
                # Thêm box cuối cùng của dòng
                boxes.append((start_x, start_y, left_lines[-1] - start_x, end_y - start_y))
            
            start_y = upper_lines[i]
    # Thêm dòng cuối cùng
    # (Đã giản lược logic đệ quy để bạn dễ hiểu syntax)

def roi_by_xy_cut(image_path):
    """Giao diện gọi thuật toán XY Cut và vẽ kết quả."""
    img = cv2.imread(str(image_path))
    if img is None: return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    boxes = []
    recursive_xy_cut(thresh, boxes)
    
    output = img.copy()
    for (x, y, w, h) in boxes:
        # Chỉ lấy các box có kích thước hợp lý
        if w > 5 and h > 5:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)
            
    print(f"[LOG] XY Cut đã phân tách được {len(boxes)} vùng văn bản.")
    return output

if __name__ == "__main__":
    sample_id = "007"
    img_filepath = DATA_RAW_DIR / "img" / f"{sample_id}.jpg"

    # Cách dùng: Gọi các hàm và so sánh
    sample_roi = masterclass_roi_extraction(sample_id)
    ccl_result = roi_by_ccl(img_filepath)
    xy_cut_result = roi_by_xy_cut(img_filepath)

    if sample_roi is not None:
        processed_roi = process_roi(sample_roi)
        
        # Hiển thị đa biểu đồ 2x2
        plt.figure(figsize=(16, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(sample_roi, cv2.COLOR_BGR2RGB))
        plt.title("1. Manual ROI (Dataset CSV)")
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(processed_roi, cmap='gray')
        plt.title("2. Processed ROI (Day 2&3 Clean)")
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(ccl_result, cv2.COLOR_BGR2RGB))
        plt.title("3. Auto: CCL (Connected Components)")
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(xy_cut_result, cv2.COLOR_BGR2RGB))
        plt.title("4. Auto: XY Cut (Projection Profile)")
        plt.axis('off')

        plt.tight_layout()
        plt.suptitle(f"Masterclass Day 5: Comparison of ROI Extraction Methods (Sample {sample_id})", fontsize=16)
        plt.show()
