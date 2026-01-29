import cv2
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Cấu hình đường dẫn, set đường dẫn để lấy thư mục tương ứng
DATA_RAW_DIR = Path("../../data/raw")
IMG_DIR = DATA_RAW_DIR / "img"
BOX_DIR = DATA_RAW_DIR / "box"
KEY_DIR = DATA_RAW_DIR / "key"

def load_sample(sample_id):
    """Đọc ảnh, box và key của một mẫu.
    đầu vào là tên ảnh-> đường dẫn ảnh -> box_path, key_path
    """
    img_path = IMG_DIR / f"{sample_id}.jpg"
    box_path = BOX_DIR / f"{sample_id}.csv"
    key_path = KEY_DIR / f"{sample_id}.json"

    if not img_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

    # Đọc ảnh
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Đọc box (x1, y1, x2, y2, x3, y3, x4, y4, [text])
    # box có dạng chứ nhật nên sẽ có 4 điểm tương ứng với 8 tọa độ là (x1, y1, x2, y2, x3, y3, x4, y4)
    # Vì text có thể chứa dấu phẩy, ta đọc thủ công từng dòng để tránh lỗi
    # đọc hết 8 tọa độ thì sẽ đến text, đưa vào list
    data = []
    with open(box_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                # 8 phần tử đầu là tọa độ, phần còn lại (nếu có) là text
                coords = [float(p) for p in parts[:8]]
                data.append(coords)
    # dataframe chứa các box
    boxes_df = pd.DataFrame(data)
    
    # Đọc nhãn văn bản
    with open(key_path, 'r', encoding='utf-8') as f:
        key_data = json.load(f)
    
    # kết quả trả về là ảnh, box và key
    return image, boxes_df, key_data

def visualize_data(sample_id="000"):
    try:
        image, boxes_df, key_data = load_sample(sample_id)
        print(f"--- Đang hiển thị mẫu: {sample_id} ---")
        print(f"Dữ liệu Keys: {json.dumps(key_data, indent=2, ensure_ascii=False)}")
        
        vis_image = image.copy()
        for index, row in boxes_df.iterrows():
            # Lấy 8 tọa độ đầu tiên và biến thành mảng (4,1,2)
            pts = row.values[:8].reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(vis_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # (Tùy chọn) Gắn ID hoặc chữ lên ảnh nếu muốn
            x, y = pts[0][0]
            cv2.putText(vis_image, str(index), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.title(f"Sample {sample_id} - {len(boxes_df)} boxes found")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    # Bạn có thể đổi sample_id thành các số khác như "001", "002"...
    visualize_data("002")
