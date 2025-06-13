# Kindle Review Sentiment Analysis using Word2Vec

This project is a web application for sentiment analysis of Kindle book reviews. It uses NLP preprocessing, Word2Vec embeddings for feature extraction, and a Random Forest classifier to predict whether a review is positive or negative. The app is built with Flask and demonstrates an end-to-end machine learning pipeline, from data cleaning to deployment.

## Features

- Cleans and preprocesses review text (lowercasing, stopword removal, lemmatization)
- Converts text to vectors using Word2Vec
- Trains a Random Forest classifier for sentiment prediction
- Flask web interface for user input and instant prediction
- Example reviews for testing

## What is Word2Vec in NLP?

Word2Vec is a popular technique in Natural Language Processing that transforms words into dense vector representations based on their context in large text corpora. Unlike traditional bag-of-words, Word2Vec captures semantic relationships, so similar words have similar vectors. This allows machine learning models to better understand the meaning and sentiment of text.

## Project Structure

```
kindle-review-sentiment-word2vec/
│
├── main.py                # Flask app for prediction
├── sentiment_model.pkl    # Pickled classifier and Word2Vec model
├── templates/
│   └── index.html         # Web UI (with embedded CSS)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── data/
    └── all_kindle_review.csv  # (Optional) Raw review data
```

## How to Run

1. Install dependencies:  
   `pip install -r requirements.txt`
2. Ensure `sentiment_model.pkl` is present (see notebook for training).
3. Run the app:  
   `python main.py`
4. Open [http://localhost:5000](http://localhost:5000) in your browser.

## Example Inputs

**Positive Review:**  
> I absolutely loved this book! The story was engaging and the characters were well developed. Highly recommended for anyone who enjoys a good mystery.

**Negative Review:**  
> The book was full of typos and the characters were boring. I regret buying this Kindle version.

---

**Enjoy exploring NLP with Word2Vec and
