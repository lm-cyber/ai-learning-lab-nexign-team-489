from fastapi import FastAPI
from app.db import Base, engine
from app.endpoints import sentiment,data

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"])
app.include_router(data.router, prefix="/data", tags=["data"])