import numpy as np
import nltk
import re
import swifter
from nltk.corpus import stopwords
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
list_stopwords = stopwords.words('indonesian')
nltk.download('punkt')


class preprocessing():
    mentah = None
    df = None

    def read_data(self):
        self.df = pd.read_csv("data/dataset.csv", sep=';')
        self.df = self.df.sample(
            frac=1.0, random_state=1).reset_index(drop=True)
        self.mentah = self.df

    def clean_punct(self):
        self.df['Kalimat'] = self.df['Kalimat'].fillna('').astype(str).str.replace(
            r'[^A-Za-z ]', '', regex=True).replace('', np.nan, regex=False)

    def lower_case(self):
        self.df = self.df.astype(str).apply(lambda x: x.str.lower())

    def tokenizing(self):
        self.df['Kalimat'] = self.df.apply(
            lambda row: nltk.word_tokenize(row['Kalimat']), axis=1)

    def stopwords_removal(self):
        [word for word in self.df['Kalimat'] if word not in list_stopwords]

    def lowers(self):
        self.df['Lowers'] = [", ".join(review)
                             for review in self.df['Kalimat'].values]

    def stemming(self):
        stemmer = factory.create_stemmer()
        self.df['Lowers'] = self.df['Lowers'].apply(
            lambda text: stemmer.stem(text))

    def save_preprocessing(self):
        self.df.to_csv("data/Text_preprocessing.csv", sep=";")
        mentah = np.array(self.mentah).tolist()
        preprocessing = np.array(self.df).tolist()

        return [mentah, preprocessing]
