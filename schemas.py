from pydantic import BaseModel

class BoundingBox(BaseModel):
    """Pydantic model cho Bounding Box."""
    x1: int
    y1: int
    x2: int
    y2: int

class Prediction(BaseModel):
    """Pydantic model cho Dự đoán."""
    label: str
    confidence: float
    bbox: BoundingBox
