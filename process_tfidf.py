import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_tfidf = TfidfVectorizer(max_features=500)


class tfidf():
    df = None
    X_tfidf = None

    def read_data(self):
        self.df = pd.read_csv("data\Text_preprocessing.csv", sep=';')

    def tfidf(self):
        self.X_tfidf = vectorizer_tfidf.fit_transform(
            self.df['Kalimat'].values.astype('U'))
        with open("model/tfidf_fit", "wb") as r:
            pickle.dump(self.X_tfidf, r)
        with open("model/tfidf", "wb") as r:
            pickle.dump(vectorizer_tfidf, r)

    def save_tfidf(self):
        tabelTFIDF = pd.DataFrame(
            self.X_tfidf.todense(), columns=vectorizer_tfidf.get_feature_names())
        tabelTFIDF.to_csv("data/TFIDF.csv", sep=";")
