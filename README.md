
# sentiment-analysis-app
A Sentiment Analysis web app with FastAPI backend and simple frontend.
=======
# Sentiment Analysis Web Application

## Project Overview

This project is a simple Sentiment Analysis web application where users can input text (e.g., tweets, reviews, messages) and get an instant sentiment classification: **Positive**, **Neutral**, or **Negative**. The app also supports batch CSV uploads for analyzing multiple texts at once and maintains a history of analyzed texts.

---

## Features

- **Single Text Input:** Type or paste text and get sentiment analysis with confidence score.
- **Batch Upload:** Upload CSV files containing texts to analyze multiple entries at once.
- **History Tracking:** See the history of all analyzed texts during the session.
- **Color-coded Results:**  
  - Green for Positive  
  - Red for Negative  
  - Grey for Neutral

---

## Technology Stack

- **Frontend:**  
  - Simple HTML, CSS, and JavaScript for UI and interaction.

- **Backend:**  
  - FastAPI as the web framework.  
  - Hugging Face Transformers pipeline using the DistilBERT model for sentiment analysis.  
  - Python libraries: `fastapi`, `uvicorn`, `transformers`, `torch`, `pydantic`, `chardet`.

---

## How to Run Locally

1. Clone the repository:  
   ```bash
   git clone <repo_url>
   cd <repo_folder>/backend
Create and activate a virtual environment (optional but recommended):
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Run the FastAPI backend server:
uvicorn app:app --reload
Open your browser and go to:
http://127.0.0.1:8000

