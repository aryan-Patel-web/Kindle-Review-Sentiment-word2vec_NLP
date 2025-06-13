from flask import Flask, render_template, request
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess

app = Flask(__name__)

# Load models
with open('sentiment_model.pkl', 'rb') as f:
    data = pickle.load(f)
    classifier = data['classifier']
    word2vec = data['word2vec']

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

def avg_word2vec(doc):
    words = simple_preprocess(doc)
    vectors = [word2vec.wv[word] for word in words if word in word2vec.wv]
    if vectors:
        return np.mean(vectors, axis=0).reshape(1, -1)
    else:
        # Return a zero vector if no words found
        return np.zeros((1, word2vec.vector_size))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    review = ''
    if request.method == 'POST':
        review = request.form['review']
        clean_review = preprocess(review)
        features = avg_word2vec(clean_review)
        pred = classifier.predict(features)[0]
        prediction = 'Positive' if pred == 1 else 'Negative'
    return render_template('index.html', prediction=prediction, review=review)

if __name__ == '__main__':
    app.run(debug=True)