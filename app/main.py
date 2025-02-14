from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from transformers import pipeline
import pandas as pd
import io

app = FastAPI()

sentiment_model = pipeline(model="seara/rubert-tiny2-russian-sentiment")


@app.post("/analyze_text")
async def analyze_text(text: str = Form(...)):
    """
    Analyze a text string and return the sentiment class and score.
    """
    try:
        result = sentiment_model(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model inference: {e}")
    return {"result": result}


@app.post("/analyze_file")
async def analyze_file(
    file: UploadFile = File(...),
    column: str = Form(...)
):
    """
    Analyze a CSV or XLS/XLSX file.
    The user must specify the column name containing text for sentiment analysis.
    Returns a JSON array with each text entry's sentiment analysis.
    """
    # Read file contents and parse according to file type
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            # For CSV, decode the bytes to string and use StringIO
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith((".xls", ".xlsx")):
            # For Excel, use BytesIO directly
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV or XLS/XLSX file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    # Check if the specified column exists
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found in the file.")

    results = []
    # Process each row in the selected column
    for text in df[column].astype(str).tolist():
        try:
            analysis = sentiment_model(text)
        except Exception as e:
            analysis = [{"error": str(e)}]
        results.append({
            "text": text,
            "result": analysis
        })

    return {"results": results}
