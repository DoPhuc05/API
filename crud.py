from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
import models, schemas
from database import SessionLocal, engine
import cv2
import numpy as np
from schemas import Prediction, BoundingBox

app = FastAPI()

# Hàm tạo session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Hàm lưu dự đoán vào cơ sở dữ liệu
def save_prediction(db: Session, prediction: Prediction):
    # Tạo Bounding Box
    bbox_db = models.BoundingBoxDB(
        x1=prediction.bbox.x1,
        y1=prediction.bbox.y1,
        x2=prediction.bbox.x2,
        y2=prediction.bbox.y2
    )
    db.add(bbox_db)
    db.commit()
    db.refresh(bbox_db)

    # Tạo Prediction
    prediction_db = models.PredictionDB(
        label=prediction.label,
        confidence=prediction.confidence,
        bbox_id=bbox_db.id
    )
    db.add(prediction_db)
    db.commit()
    db.refresh(prediction_db)

    return prediction_db

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Nhận ảnh từ người dùng, chạy YOLOv8 và lưu kết quả vào cơ sở dữ liệu"""
    image_bytes = await file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Chạy mô hình YOLOv8
    results = model(source=image)
    predictions = []

    for result in results:
        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
            score = result.boxes.conf[i].item()
            label = int(result.boxes.cls[i].item())

            prediction = prediction(
                label=model.names[label],
                confidence=round(score, 2),
                bbox=BoundingBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
            )

            # Lưu vào cơ sở dữ liệu
            save_prediction(db, prediction)
            predictions.append(prediction)

    return {"predictions": predictions}
