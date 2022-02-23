import re
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
factory = StemmerFactory()
stemmer = factory.create_stemmer()


class detect():
    text = None

    def lower_case(self):
        self.text = self.text.lower()

    def clean_punct(self):
        self.text = re.compile('[/(){}\[\]\|@,;]').sub('', self.text)
        self.text = re.compile('[^0-9a-z]').sub(' ', self.text)

    def stopwords_removal(self):
        data = StopWordRemoverFactory().get_stop_words()
        dictionary = ArrayDictionary(data)
        stopword = StopWordRemover(dictionary)
        self.text = stopword.remove(self.text)

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
