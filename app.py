from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class Review(BaseModel):
    text:str

app = FastAPI(title="Sentiment analysis on IMDB", version="1.0")

model_path = "models/imdb_model"

tokeniser = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokeniser)


@app.post("/predict")
async def predict_sentiment(review:Review):
    result = sentiment_pipeline(review.text)[0]
    return {"sentiment:": result["label"].lower(),
            "confidence:":float(result['score'])}