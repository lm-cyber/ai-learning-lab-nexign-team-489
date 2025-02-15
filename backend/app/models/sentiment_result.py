from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.db import Base

class SentimentResult(Base):
    __tablename__ = "sentiment_results"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    result = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    true_result = Column(String, nullable=True)  # Field for later validation
    created_at = Column(DateTime(timezone=True), server_default=func.now())
