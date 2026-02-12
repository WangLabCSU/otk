from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from .config import DATABASE_URL

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    original_filename = Column(String)
    uploaded_file_path = Column(String)
    results_file_path = Column(String, nullable=True)
    
    model_used = Column(String, nullable=True)
    device_used = Column(String, nullable=True)
    
    total_rows = Column(Integer, nullable=True)
    total_samples = Column(Integer, nullable=True)
    total_genes = Column(Integer, nullable=True)
    
    processing_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    validation_report = Column(JSON, nullable=True)
    
    priority = Column(Integer, default=0)

class Statistics(Base):
    __tablename__ = "statistics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, default=datetime.utcnow)
    
    total_jobs = Column(Integer, default=0)
    completed_jobs = Column(Integer, default=0)
    failed_jobs = Column(Integer, default=0)
    
    total_rows_processed = Column(Integer, default=0)
    total_samples_processed = Column(Integer, default=0)
    
    avg_processing_time = Column(Float, default=0.0)
    
    gpu_jobs = Column(Integer, default=0)
    cpu_jobs = Column(Integer, default=0)
    
    validation_errors = Column(Integer, default=0)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
