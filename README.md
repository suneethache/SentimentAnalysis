# 🧠 Lightweight LLM Sentiment Analysis on customer reviews and feedback

This project demonstrates an end-to-end pipeline for fine-tuning a lightweight language model (DistilBERT) on the review dataset for sentiment analysis using **LoRA (Low-Rank Adaptation)**. 

I am using IMDB dataset to demonstrate the real usecase (Since I am not allowed to share the dataset), which is adaptable to proprietary business data sources.

It is optimized for CPU training and includes model deployment with **FastAPI**, experiment tracking via **MLflow**, and a complete **CI/CD pipeline with GitHub Actions**.

---

## 📌 Features

- ✅ Fine-tuning with Hugging Face Transformers and PEFT (LoRA)
- ✅ Efficient training on CPU (no GPU required)
- ✅ Inference-ready REST API built with FastAPI
- ✅ Dockerized for portable deployment
- ✅ MLflow integration for experiment tracking and reproducibility
- ✅ GitHub Actions CI/CD pipeline for automated testing and Docker builds

---

## 🧪 Dataset

- You can also use your own dataset to finetune the model.
- I have used IMDB reviews dataset from Hugging face library to make things easier to explain the workflow.
- But when you use custom dataset make sure the token size matches the distilBert input token limit.

- **IMDB Movie Reviews**  
  - 50,000 reviews (25k train / 25k test)
  - Binary classification (positive / negative sentiment)

---

## 📁 Project Structure

imdb-sentiment-llm/
├── train.py
├── app.py
├── Dockerfile
├── requirements.txt
├── models/
│   └── imdb_model/        # fine-tuned model files (or you can download in app)
└── .github/
    └── workflows/
        └── ci-cd.yml      # GitHub Actions workflow


## Running the code

# Install dependencies

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

By running the above lines of code, everything that needed to train/ test the model will be installed.

** ## 2. To finetune **
```bash
python train.py

## Run FastAPI inference server
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Testing the API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The movie was absolutely fantastic!"}'

###Docker Deployment

## Build the Docker image
```bash
docker build -t imdb-sentiment-app

## Run the container
```bash
docker run -p 8000:8000 imdb-sentiment-app


### MLflow tracking
```bash

mlflow ui




