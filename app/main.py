from flask import Flask, render_template, request, jsonify
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import pickle
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS

clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z]')


def clean_punct(text):
    text = clean_spcl.sub('', text)
    text = clean_symbol.sub(' ', text)
    return text


factory = StemmerFactory()
stemmer = factory.create_stemmer()


def stemmed_wrapper(term):
    return stemmer.stem(term)


app = Flask(__name__)
df = pd.read_csv("dataset\Text_Preprocessing.csv")
vectorizer_tfidf = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer_tfidf.fit_transform(df['Lowers'])
# tfidf = pickle.load(open("tfidf", 'rb'))

with open("modelSVM\model", "rb") as r:
    model = pickle.load(r)

CORS(app)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        try:
            quest = request.json.get('text')
            Input = clean_punct(quest)
            Input = Input.lower()
            Input = stemmed_wrapper(Input)
            Input = vectorizer_tfidf.transform([Input])
            prediction = model.predict(np.array(Input).tolist()).tolist()
            return ({
                'pertanyaan': quest,
                'predict': prediction[0]
            })
        except:
            return jsonify({'predict': 'Tidak ada inputan'})
    else:
        return "<h1>Welcome to OOD API</h1>"

# if __name__ == "__main__":
#     app.debug = True
#     app.run(host='0.0.0.0', port="5000")
