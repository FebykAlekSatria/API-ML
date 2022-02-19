# from sklearn.feature_extraction.text import TfidfVectorizer
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
    clean_spcl = re.compile('[/(){}\[\]\|@,;]')
    clean_symbol = re.compile('[^0-9a-z]')

    def lower_case(self):
        self.text = self.text.lower()

    def clean_punct(self):
        self.text = self.clean_spcl.sub('', self.text)
        self.text = self.clean_symbol.sub(' ', self.text)

    def stopwords_removal(self):
        return [word for word in self.text if word not in list_stopwords]

    def stemmed_wrapper(self):
        self.text = self.stemmer.stem(self.text)

    def predict(self):
        with open("model/tfidf", "rb") as r:
            vectorizer_tfidf = pickle.load(r)
        Input = vectorizer_tfidf.transform([self.text])
        with open("model/svm", "rb") as r:
            model = pickle.load(r)
        prediction = model.predict(Input)
        return prediction
