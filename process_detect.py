import numpy as np
import nltk
import re
import swifter
import pickle
from nltk.corpus import stopwords
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
list_stopwords = stopwords.words('indonesian')


class detect():
    text = None

    def lower_case(self):
        self.text = self.text.lower()

    def clean_punct(self):
        self.text = re.compile('[/(){}\[\]\|@,;]').sub('', self.text)
        self.text = re.compile('[^0-9a-z]').sub(' ', self.text)

    def stopwords_removal(self):
        return [word for word in self.text if word not in list_stopwords]

    def stemming(self):
        self.text = stemmer.stem(self.text)

    def predict(self):
        with open("model/tfidf", "rb") as r:
            vectorizer_tfidf = pickle.load(r)
        Input = vectorizer_tfidf.transform([self.text])
        with open("model/svm", "rb") as r:
            model = pickle.load(r)
        prediction = model.predict(Input)
        return prediction
