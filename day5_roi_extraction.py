import cv2
import pandas as pd
import os
from pathlib import Path

# Cấu hình
DATA_RAW_DIR = Path("../../data/raw")
SAVE_DIR = Path("../../data/processed/week1_crops")

def extract_rois(sample_id="000"):
    img_path = DATA_RAW_DIR / "img" / f"{sample_id}.jpg"
    box_path = DATA_RAW_DIR / "box" / f"{sample_id}.csv"
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    image = cv2.imread(str(img_path))
    
    # Đọc box thủ công để tránh lỗi dấu phẩy trong text
    data = []
    with open(box_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                coords = [float(p) for p in parts[:8]]
                data.append(coords)
    
    boxes = pd.DataFrame(data)

    print(f"Bắt đầu cắt {len(boxes)} vùng từ ảnh {sample_id}...")

    for i, row in boxes.iterrows():
        # Lấy tọa độ (giải sử x1, y1, x2, y2, x3, y3, x4, y4)
        # Để đơn giản, ta lấy [min_x, min_y, max_x, max_y]
        pts = row.values[:8].reshape((-1, 2))
        x_min, y_min = pts.min(axis=0).astype(int)
        x_max, y_max = pts.max(axis=0).astype(int)

        # Cắt ảnh (Crop) - Lưu ý giới hạn biên ảnh
        roi = image[max(0, y_min):y_max, max(0, x_min):x_max]
        
        if roi.size > 0:
            save_path = SAVE_DIR / f"{sample_id}_box_{i}.jpg"
            cv2.imwrite(str(save_path), roi)

    print(f"Hoàn thành! Ảnh được lưu tại: {SAVE_DIR}")

if __name__ == "__main__":
    extract_rois("000")
