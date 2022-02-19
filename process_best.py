from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle
import pandas as pd


class best_params():
    df = None
    y = None
    X = None
    parameters = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'C': [0.5, 1, 10, 100],
        'gamma': ['scale', 1, 0.1, 0.01, 0.001]
    }

    def read_data(self):
        self.df = pd.read_csv("data/Text_preprocessing.csv", sep=';')
        with open("model/tfidf_fit", "rb") as r:
            vectorizer_tfidf = pickle.load(r)
        self.y = self.df.Kelas
        self.X = vectorizer_tfidf

    def best_params(self):
        grid_search = GridSearchCV(estimator=svm.SVC(random_state=0),
                                   param_grid=self.parameters,
                                   n_jobs=6,
                                   verbose=1,
                                   scoring='accuracy'
                                   )

        grid_search.fit(self.X, self.y)

        best_params = grid_search.best_estimator_.get_params()

        for param in self.parameters:
            param: best_params[param]
        return best_params
