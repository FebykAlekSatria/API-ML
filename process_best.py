from sklearn.metrics import confusion_matrix
import time
from sklearn.model_selection import KFold
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm
import pandas as pd


class best_params():
    df = None
    X = None
    Y = None
    kernel = ['linear', 'rbf']
    c = [0.1, 1, 10,]
    best_kernel=None
    best_c=None
    accurasy = 0

    def read_data(self):
        self.df = pd.read_csv("data/Text_preprocessing.csv", sep=';')
        with open("model/tfidf_fit", "rb") as r:
            vectorizer_tfidf = pickle.load(r)
        self.Y = self.df.Kelas
        self.X = vectorizer_tfidf

    def best_params(self):
        for kernel in self.kernel:
            for c in self.c:
                Y = self.Y
                X = self.X
                model = svm.SVC(kernel=kernel, C=c)
                mean=0
                cv = KFold(n_splits=10, shuffle=True, random_state=0)
                for train_index, test_index in cv.split(X):
                    X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
                    model.fit(X_train, Y_train)
                    Y_pred = model.predict(X_test)
                    acc = round(accuracy_score(Y_test, Y_pred), 3)
                    mean = mean + acc
                    

                mean = mean/10 

                if mean > self.accurasy:
                    self.accurasy = mean
                    self.best_kernel = kernel
                    self.best_c = c

        return self.accurasy, self.best_kernel, self.best_c
