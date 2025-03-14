from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey  # Thêm ForeignKey vào đây
from sqlalchemy.orm import relationship
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Class cho bảng BoundingBox
class BoundingBox(Base):
    __tablename__ = 'bounding_boxes'
    
    id = Column(Integer, primary_key=True)
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    
    # Quan hệ với bảng PredictionDB
    predictions = relationship('PredictionDB', back_populates='bounding_box')

# Class cho bảng PredictionDB
class PredictionDB(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    label = Column(String)
    confidence = Column(Float)
    
    # ForeignKey trỏ đến bảng BoundingBox
    bbox_id = Column(Integer, ForeignKey('bounding_boxes.id'))
    
    # Quan hệ với bảng BoundingBox
    bounding_box = relationship('BoundingBox', back_populates='predictions')

# Thiết lập kết nối với cơ sở dữ liệu SQLite (hoặc cơ sở dữ liệu khác)
engine = create_engine('sqlite:///predictions.db')

# Tạo bảng trong cơ sở dữ liệu
Base.metadata.create_all(engine)
