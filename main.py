import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO  
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pyngrok import ngrok
import uvicorn
from schemas import Prediction, BoundingBox
import aiofiles

# Khởi tạo FastAPI
app = FastAPI()

# Cấu hình Ngrok
NGROK_AUTH_TOKEN = "2tcouva4KHG2fccLtZPW7PDXMvZ_4YCgrCFDUKea2cJUhYj8t"  # 🔥 Thay bằng token của bạn
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Đường dẫn đến mô hình YOLOv8
MODEL_DIR = "/content/models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "best (3).pt")

# Kiểm tra mô hình
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy file {MODEL_PATH}")

# Load mô hình YOLOv8
print(f"🔄 Đang tải mô hình YOLOv8 từ {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("✅ Mô hình YOLOv8 đã sẵn sàng!")

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """Nhận ảnh từ người dùng, chạy YOLOv8 và trả về kết quả JSON"""
    # Đọc file ảnh từ request
    image_bytes = await file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Chạy mô hình YOLOv8
    results = model(source=image)

    # Lấy danh sách vật thể nhận diện được
    predictions = []
    for result in results:
        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
            score = result.boxes.conf[i].item()
            label = int(result.boxes.cls[i].item())

            # Tạo đối tượng Prediction
            prediction = Prediction(
                label=model.names[label],
                confidence=round(score, 2),
                bbox=BoundingBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
            )
            predictions.append(prediction)

    return JSONResponse({"predictions": predictions})

@app.post("/predict-image/")
async def predict_and_return_image(file: UploadFile = File(...)):
    """Nhận ảnh từ người dùng, chạy YOLOv8 và trả về ảnh có bounding box"""
    # Đọc file ảnh từ request
    image_bytes = await file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Chạy mô hình YOLOv8 và vẽ bounding boxes
    results = model(source=image)
    for result in results:
        image_with_boxes = result.plot()  # Vẽ lên ảnh

    # Lưu ảnh kết quả
    output_path = "output.jpg"
    cv2.imwrite(output_path, image_with_boxes)

    # Trả về ảnh đã xử lý
    return FileResponse(output_path, media_type="image/jpeg")

@app.post("/predict-video/")
async def predict_video(file: UploadFile = File(...)):
    """Nhận video từ người dùng, xử lý từng 5 frame và trả về video có bounding box"""

    # Kiểm tra định dạng file
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        return JSONResponse({"error": "Chỉ hỗ trợ file video (.mp4, .avi, .mov)!"}, status_code=400)

    # Ghi file video đầu vào
    input_video_path = "temp_input.mp4"
    async with aiofiles.open(input_video_path, "wb") as f:
        await f.write(await file.read())

    # Kiểm tra file đã lưu thành công chưa
    if not os.path.exists(input_video_path):
        return JSONResponse({"error": "Không thể lưu file video!"}, status_code=500)

    # Mở video bằng OpenCV
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return JSONResponse({"error": "Không thể mở video! Kiểm tra lại file hoặc codec."}, status_code=400)

    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tạo bộ ghi video đầu ra
    output_video_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Hết video

        frame_count += 1

        # Chỉ xử lý mỗi 5 frame
        if frame_count % 5 == 0:
            results = model(frame)
            if results and len(results) > 0:
                frame = results[0].plot()  # plot() thay thế render() trong YOLOv8
        # Ghi frame vào video đầu ra
        out.write(frame)

    # Đóng video
    cap.release()
    out.release()
    os.remove(input_video_path)  # Xóa video gốc sau khi xử lý

    # Trả về video đã xử lý
    return FileResponse(output_video_path, media_type="video/mp4", filename="processed_video.mp4")

# Khởi chạy FastAPI + Ngrok
def start_ngrok():
    public_url = ngrok.connect(8000).public_url
    print(f"🔥 Ngrok URL: {public_url}")

if __name__ == "__main__":
    start_ngrok()  # Chạy Ngrok
    uvicorn.run(app, host="0.0.0.0", port=8000)