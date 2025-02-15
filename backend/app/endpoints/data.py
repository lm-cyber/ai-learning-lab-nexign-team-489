from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
import pandas as pd
import io
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from sqlalchemy import func


from app.db import SessionLocal
from app.models.sentiment_result import SentimentResult
router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


    
@router.get("/results", response_model=List[dict])
async def get_results(db: Session = Depends(get_db)):
    """
    Retrieve and return all SentimentResult records.
    """
    results = db.query(SentimentResult).all()
    return [
        {
            "id": r.id,
            "text": r.text,
            "result": r.result,
            "score": r.score,
            "true_result": r.true_result,
            "created_at": r.created_at
        }
        for r in results
    ]



@router.put("/validate/{result_id}")
async def validate_result(result_id: int, true_result: str, db: Session = Depends(get_db)):
    """
    Update the 'true_result' field of a SentimentResult record.
    """
    sentiment_result = db.query(SentimentResult).filter(SentimentResult.id == result_id).first()
    if not sentiment_result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    sentiment_result.true_result = true_result
    db.commit()
    db.refresh(sentiment_result)
    
    return {
        "id": sentiment_result.id,
        "text": sentiment_result.text,
        "result": sentiment_result.result,
        "score": sentiment_result.score,
        "true_result": sentiment_result.true_result,
        "created_at": sentiment_result.created_at
    }



@router.get("/result/{result_id}", response_model=dict)
async def get_result_by_id(result_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a single SentimentResult record by its ID.
    """
    sentiment_result = db.query(SentimentResult).filter(SentimentResult.id == result_id).first()
    if not sentiment_result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return {
        "id": sentiment_result.id,
        "text": sentiment_result.text,
        "result": sentiment_result.result,
        "score": sentiment_result.score,
        "true_result": sentiment_result.true_result,
        "created_at": sentiment_result.created_at
    }


@router.get("/results/date", response_model=List[dict])
async def get_results_by_date(date: str = Query(..., description="Date in YYYY-MM-DD format"), db: Session = Depends(get_db)):
    """
    Retrieve all SentimentResult records created on a given date.
    The date should be provided in the query parameter (YYYY-MM-DD).
    """
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    results = db.query(SentimentResult).filter(func.date(SentimentResult.created_at) == target_date).all()
    
    return [
        {
            "id": r.id,
            "text": r.text,
            "result": r.result,
            "score": r.score,
            "true_result": r.true_result,
            "created_at": r.created_at
        }
        for r in results
    ]