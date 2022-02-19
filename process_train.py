from sklearn.metrics import confusion_matrix
import time
from sklearn.model_selection import KFold
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm
import pandas as pd


class training():
    df = None
    X = None
    Y = None
    kernel = None
    c = None
    gamma = None

    def read_data(self):
        self.df = pd.read_csv("data/Text_preprocessing.csv", sep=';')
        with open("model/tfidf_fit", "rb") as r:
            vectorizer_tfidf = pickle.load(r)
        self.Y = self.df.Kelas
        self.X = vectorizer_tfidf

    def classify(self):
        Y = self.Y
        X = self.X
        model = svm.SVC(kernel=self.kernel, C=self.c, gamma=self.gamma)
        modelSVM = svm.SVC(kernel=self.kernel, C=self.c, gamma=self.gamma)
        modelSVM.fit(X, Y)
        with open("model/svm", "wb") as r:
            pickle.dump(modelSVM, r)
        scores = []
        confusion_score = []
        scores.append(['Uji ke', 'akurasi', 'precision',
                       'recall', 'f- measure', 'waktu Komputasi'])
        cv = KFold(n_splits=10)
        index = 1
        for train_index, test_index in cv.split(X):
            X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
            start_time = time.time()
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            [tn, fp], [fn, tp] = confusion_matrix(Y_test, Y_pred)

            acc = round(accuracy_score(Y_test, Y_pred), 4)
            pre = round(precision_score(Y_test, Y_pred, pos_label='ood'), 4)
            rec = round(recall_score(Y_test, Y_pred, pos_label='ood'), 4)
            f1 = round(f1_score(Y_test, Y_pred, pos_label='ood'), 4)
            computasion_time = round((time.time() - start_time), 4)
            scores.append([index, acc, pre, rec, f1, computasion_time])
            confusion_score.append([tn, fp, fn, tp])
            index += 1

        temp = ['Rata-rata', 0, 0, 0, 0, 0]
        for i in range(1, 11):
            for j in range(1, 6):
                temp[j] += scores[i][j]

        for i in range(1, 6):
            temp[i] = round((temp[i]/10), 4)

        scores.append(temp)
        scores.append(confusion_score)
        return scores
