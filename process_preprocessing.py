import numpy as np
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
factory = StemmerFactory()
stemmer = factory.create_stemmer()


class preprocessing():
    row = None
    df = None

    def read_data(self):
        self.df = pd.read_csv("data/dataset.csv", sep=';')
        self.row = self.df

    def clean_punct(self):
        self.df['Kalimat'] = self.df['Kalimat'].fillna('').astype(str).str.replace(
            r'[^A-Za-z ]', '', regex=True).replace('', np.nan, regex=False)

    def lower_case(self):
        self.df = self.df.astype(str).apply(lambda x: x.str.lower())

    def stopwords_removal(self):
        data = StopWordRemoverFactory().get_stop_words()
        dictionary = ArrayDictionary(data)
        stopword = StopWordRemover(dictionary)
        self.df['Kalimat'] = self.df['Kalimat'].apply(
            lambda text: stopword.remove(text))

    def stemming(self):
        stemmer = factory.create_stemmer()
        self.df['Kalimat'] = self.df['Kalimat'].apply(
            lambda text: stemmer.stem(text))

    def save_preprocessing(self):
        self.df.to_csv("data/Text_preprocessing.csv", sep=";")
        row = np.array(self.row).tolist()
        preprocessing = np.array(self.df).tolist()

        return [row, preprocessing]
