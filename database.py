from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from models import Base

SQLALCHEMY_DATABASE_URL = "sqlite:///predictions.db"  # Bạn có thể thay thế bằng URL của cơ sở dữ liệu của bạn

# Tạo engine và session
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Tạo các bảng
Base.metadata.create_all(bind=engine)
