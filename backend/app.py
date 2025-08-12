
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline
import csv
import io
import chardet
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from /static
app.mount("/static", StaticFiles(directory="static"), name="static")

class TextData(BaseModel):
    text: str

sentiment_pipeline = None
history = []  # store analyzed texts
positive_words = set()
negative_words = set()

executor = ThreadPoolExecutor(max_workers=4)  # For async pipeline calls

def load_word_list(file_path):
    words = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(';'):
                    words.add(line.lower())
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(';'):
                    words.add(line.lower())
    return words

@app.on_event("startup")
def startup_event():
    global sentiment_pipeline, positive_words, negative_words
    sentiment_pipeline = pipeline("sentiment-analysis")
    positive_words = load_word_list("lexicon/positive-words.txt")
    negative_words = load_word_list("lexicon/negative-words.txt")

def basic_sentiment_logic(text: str):
    text_lower = text.lower()
    tokens = set(text_lower.split())
    # Neutral / ambiguous checks
    if any(kw in text_lower for kw in ["neutral", "okay", "fine", "so-so", "don't know", "not sure", "maybe"]):
        return "Neutral", 80.0
    has_pos = bool(tokens.intersection(positive_words))
    has_neg = bool(tokens.intersection(negative_words))
    # Negation priority
    if "not good" in text_lower:
        return "Negative", 95.0
    if "not bad" in text_lower:
        return "Positive", 90.0
    if "not sad" in text_lower:
        return "Positive", 90.0
    if has_pos and not has_neg:
        return "Positive", 90.0
    if has_neg and not has_pos:
        return "Negative", 90.0
    return None, None  # fallback to model

async def async_sentiment_analysis(text: str):
    loop = asyncio.get_event_loop()
    # Run pipeline call in threadpool to avoid blocking
    result = await loop.run_in_executor(executor, sentiment_pipeline, text)
    return result[0] if result else None

@app.post("/analyze")
async def analyze_sentiment(data: TextData):
    if sentiment_pipeline is None:
        return {"error": "Model not ready"}, 503

    sentiment, confidence = basic_sentiment_logic(data.text)
    if sentiment is None:
        # fallback to model
        result = await async_sentiment_analysis(data.text)
        if result is None:
            return {"error": "Sentiment analysis failed"}, 500
        raw_label = result.get("label", "").upper()
        score = result.get("score", 0)
        if score < 0.6:
            sentiment = "Neutral"
        else:
            if raw_label == "POSITIVE":
                sentiment = "Positive"
            elif raw_label == "NEGATIVE":
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
        confidence = round(score * 100, 2)

    # Save history
    history.append({
        "text": data.text,
        "sentiment": sentiment,
        "confidence": confidence
    })

    return {"sentiment": sentiment, "confidence": confidence}

@app.post("/batch_analyze")
async def batch_analyze(file: UploadFile = File(...)):
    if sentiment_pipeline is None:
        return {"error": "Model not ready"}, 503

    content = await file.read()
    detected = chardet.detect(content)
    encoding = detected['encoding'] or 'utf-8'

    try:
        decoded = content.decode(encoding)
    except UnicodeDecodeError:
        decoded = content.decode('latin-1')

    csv_reader = csv.reader(io.StringIO(decoded))
    texts = [row[0].strip() for row in csv_reader if row and row[0].strip()]

    results = []
    # Prepare async tasks for pipeline calls for texts needing fallback
    tasks = []
    fallback_indices = []

    for i, text in enumerate(texts):
        sentiment, confidence = basic_sentiment_logic(text)
        if sentiment is not None:
            # Known sentiment from lexicon/logic
            results.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence
            })
        else:
            # Will need model call fallback - placeholder for now
            results.append(None)
            fallback_indices.append(i)
            tasks.append(async_sentiment_analysis(text))

    # Await all model calls concurrently
    model_results = await asyncio.gather(*tasks)

    # Fill results for fallback indices
    for idx, model_result in zip(fallback_indices, model_results):
        text = texts[idx]
        if model_result is None:
            sentiment = "Neutral"
            confidence = 0.0
        else:
            raw_label = model_result.get("label", "").upper()
            score = model_result.get("score", 0)
            if score < 0.6:
                sentiment = "Neutral"
            else:
                if raw_label == "POSITIVE":
                    sentiment = "Positive"
                elif raw_label == "NEGATIVE":
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
            confidence = round(score * 100, 2)

        results[idx] = {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence
        }

    # Save all to history
    for res in results:
        history.append(res)

    return {"results": results}

@app.get("/history")
def get_history():
    return {"history": history}

@app.get("/")
def index():
    return FileResponse("static/index.html")
