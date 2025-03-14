import os
import torch
from ultralytics import YOLO

# 🔥 Thiết lập đường dẫn đến API Key Kaggle
KAGGLE_CONFIG_DIR = os.path.join(os.getcwd(), "kaggle.json")  # Lưu kaggle.json trong thư mục dự án
os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()

# 🔥 Dataset chứa mô hình trên Kaggle
KAGGLE_DATASET = "tamtamne/ver2-38-8"

# 🔥 Thư mục lưu trữ mô hình
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 🔥 Kiểm tra nếu mô hình đã tồn tại thì không tải lại
MODEL_PATH = os.path.join(MODEL_DIR, "best(3).pt")

if not os.path.exists(MODEL_PATH):
    print("📥 Đang tải mô hình từ Kaggle...")
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {MODEL_DIR} --unzip")

# 🔥 Kiểm tra file tải về
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy file {MODEL_PATH}")

# 🔥 Load mô hình YOLOv8
print(f"🔄 Đang tải mô hình YOLOv8 từ {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Mô hình YOLOv8 đã sẵn sàng!")
except Exception as e:
    raise RuntimeError(f"❌ Lỗi khi tải mô hình: {e}")
