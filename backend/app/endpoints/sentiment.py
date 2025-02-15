from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
import pandas as pd
import io
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from sqlalchemy import func


from app.db import SessionLocal
from app.models.sentiment_result import SentimentResult
from ..ml_models.sentiment_model import model
router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/analyze_text")
async def analyze_text(text: str = Form(...), db: Session = Depends(get_db)):
    try:
        result,text = model(text)
        label = result[0]["label"]
        score = result[0]["score"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model inference: {e}")
    
    sentiment_result = SentimentResult(
        text=text,
        result=label,
        score=score
    )
    db.add(sentiment_result)
    db.commit()
    db.refresh(sentiment_result)
    
    return {"result": result, "db_id": sentiment_result.id}


@router.post("/analyze_file")
async def analyze_file(
    file: UploadFile = File(...),
    column: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV or XLS/XLSX file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found in the file.")

    results = []
    for text in df[column].astype(str).tolist():
        try:
            analysis, text = model(text)
            label = analysis[0]["label"]
            score = analysis[0]["score"]
        except Exception as e:
            analysis = [{"error": str(e)}]
            label = ""
            score = 0.0
        
        sentiment_result = SentimentResult(
            text=text,
            result=label,
            score=score
        )
        db.add(sentiment_result)
        db.commit()
        db.refresh(sentiment_result)
        
        results.append({
            "text": text,
            "result": analysis,
            "db_id": sentiment_result.id
        })

    return {"results": results}

