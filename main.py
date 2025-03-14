import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO  
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import aiofiles
from load_model import model  

# Khởi tạo FastAPI
app = FastAPI()

# ======================= XỬ LÝ ẢNH =======================
@app.post("/predict-image/")
async def predict_and_return_image(file: UploadFile = File(...)):
    """Nhận ảnh từ người dùng, chạy YOLOv8, vẽ bounding box & trả về ảnh"""
    image_bytes = await file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Chạy mô hình YOLOv8
    results = model(image)

    # Vẽ bounding boxes
    image_with_boxes = results[0].plot()
    output_path = "output.jpg"
    cv2.imwrite(output_path, image_with_boxes)

    return FileResponse(output_path, media_type="image/jpeg")

# ======================= XỬ LÝ VIDEO =======================
@app.post("/predict-video/")
async def predict_video(file: UploadFile = File(...)):
    """Nhận video từ người dùng, xử lý từng 5 frame và trả về video có bounding box"""

    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        return JSONResponse({"error": "Chỉ hỗ trợ file video (.mp4, .avi, .mov)!"}, status_code=400)

    input_video_path = "temp_input.mp4"
    async with aiofiles.open(input_video_path, "wb") as f:
        await f.write(await file.read())

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return JSONResponse({"error": "Không thể mở video!"}, status_code=400)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 == 0:
            results = model(frame)
            frame = results[0].plot()

        out.write(frame)

    cap.release()
    out.release()
    os.remove(input_video_path)

    return FileResponse(output_video_path, media_type="video/mp4", filename="processed_video.mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
