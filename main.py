from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize the sentiment analysis pipeline once (loads model)
sentiment_pipeline = pipeline("sentiment-analysis")

app = FastAPI()

# Define input data model
class TextIn(BaseModel):
    text: str

# Define output data model
class SentimentOut(BaseModel):
    label: str
    score: float

@app.post("/predict-sentiment/", response_model=SentimentOut)
def predict_sentiment(data: TextIn):
    result = sentiment_pipeline(data.text)[0]
    return SentimentOut(label=result['label'], score=result['score'])
