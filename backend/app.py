
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import csv
import io
import chardet
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

app = FastAPI()

# Enable CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body schema for single text analyze
class TextData(BaseModel):
    text: str

# Globals
sentiment_pipeline = None
positive_words = set()
negative_words = set()
history = []
executor = ThreadPoolExecutor(max_workers=4)

def load_word_list(file_path):
    words = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(';'):  # ignore comments in lexicon
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
    lex_path = os.path.join(os.path.dirname(__file__), "lexicon")
    positive_words.update(load_word_list(os.path.join(lex_path, "positive-words.txt")))
    negative_words.update(load_word_list(os.path.join(lex_path, "negative-words.txt")))


def lexicon_sentiment(text: str):
    text_lower = text.lower()
    tokens = text_lower.split()

    positive_indices = [i for i, t in enumerate(tokens) if t in positive_words]
    negative_indices = [i for i, t in enumerate(tokens) if t in negative_words]

    def is_negated(index):
        # Check if "not", "no", or "never" is immediately before the word (simple negation)
        if index == 0:
            return False
        return tokens[index - 1] in {"not", "no", "never"}

    pos_count = 0
    neg_count = 0

    # Count positives with negation flipping sentiment
    for i in positive_indices:
        if is_negated(i):
            neg_count += 1
        else:
            pos_count += 1

    # Count negatives with negation flipping sentiment
    for i in negative_indices:
        if is_negated(i):
            pos_count += 1
        else:
            neg_count += 1

    # Decide final sentiment based on counts
    if pos_count > neg_count:
        return "Positive", 90.0
    elif neg_count > pos_count:
        return "Negative", 90.0
    elif pos_count == neg_count and pos_count > 0:
        # If equal counts but non-zero, consider neutral
        return "Neutral", 50.0

    # No sentiment words found
    return None, None


# Async wrapper for pipeline call
async def async_sentiment_analysis(text: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, sentiment_pipeline, text)
    return result[0] if result else None

# Single text analyze endpoint
@app.post("/analyze")
async def analyze_sentiment(data: TextData):
    if sentiment_pipeline is None:
        return {"error": "Model not ready"}, 503

    sentiment, confidence = lexicon_sentiment(data.text)
    if sentiment is None:
        result = await async_sentiment_analysis(data.text)
        if result is None:
            return {"error": "Sentiment analysis failed"}, 500
        sentiment = result.get("label", "Neutral").capitalize()
        confidence = round(result.get("score", 0) * 100, 2)

    history.append({"text": data.text, "sentiment": sentiment, "confidence": confidence})

    return {"sentiment": sentiment, "confidence": confidence}

# Batch CSV analyze endpoint
@app.post("/batch_analyze")
async def batch_analyze(file: UploadFile = File(...)):
    if sentiment_pipeline is None:
        return {"error": "Model not ready"}, 503

    content = await file.read()
    encoding = chardet.detect(content)['encoding'] or 'utf-8'
    try:
        decoded = content.decode(encoding)
    except UnicodeDecodeError:
        decoded = content.decode('latin-1')

    csv_reader = csv.reader(io.StringIO(decoded))
    texts = [row[0].strip() for row in csv_reader if row and row[0].strip()]

    results = []
    tasks = []
    fallback_indices = []

    # Run lexicon sentiment first; if none, queue model analysis
    for i, text in enumerate(texts):
        sentiment, confidence = lexicon_sentiment(text)
        if sentiment is not None:
            results.append({"text": text, "sentiment": sentiment, "confidence": confidence})
        else:
            results.append(None)  # placeholder
            fallback_indices.append(i)
            tasks.append(async_sentiment_analysis(text))

    model_results = await asyncio.gather(*tasks)

    for idx, res in zip(fallback_indices, model_results):
        text = texts[idx]
        if res is None:
            sentiment = "Neutral"
            confidence = 0.0
        else:
            sentiment = res.get("label", "Neutral").capitalize()
            confidence = round(res.get("score", 0) * 100, 2)
        results[idx] = {"text": text, "sentiment": sentiment, "confidence": confidence}

    history.extend(results)

    return {"results": results}

# History endpoint (last 50 entries)
@app.get("/history")
def get_history():
    return {"history": history[-50:]}
