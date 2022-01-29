from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import time
import swifter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
import numpy as np
import pickle
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS
from nltk.corpus import stopwords
from flask import send_from_directory
from werkzeug.utils import secure_filename


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


term_dict = {}


def get_stemmed_term(document):
    return [term_dict[term] for term in document]


list_stopwords = stopwords.words('indonesian')


def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]


vectorizer_tfidf = TfidfVectorizer(max_features=500)


def tfidf():
    df = pd.read_csv("Text_prepocessing.csv", sep=';')
    X_tfidf = vectorizer_tfidf.fit_transform(df['Lowers'])
    tabelTFIDF = pd.DataFrame(
        X_tfidf.todense(), columns=vectorizer_tfidf.get_feature_names())
    tabelTFIDF.to_csv("TFIDF.csv", sep=";")

# BEST PARAMS


def best_params(df):
    y = df.Kelas
    X = vectorizer_tfidf.fit_transform(df['Lowers'])
    parameters = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'C': [0.5, 1, 10, 100],
        'gamma': ['scale', 1, 0.1, 0.01, 0.001]
    }

    grid_search = GridSearchCV(estimator=svm.SVC(random_state=0),
                               param_grid=parameters,
                               n_jobs=6,
                               verbose=1,
                               scoring='accuracy'
                               )

    grid_search.fit(X, y)

    best_params = grid_search.best_estimator_.get_params()

    for param in parameters:
        param: best_params[param]

    print(param)
    return best_params

# KLASIFIKASI SVM


# def model(df, kernel, c, gamma):
#     Y = df.Kelas
#     X = vectorizer_tfidf.fit_transform(df['Lowers'])
#     modelSVM = svm.SVC(kernel=kernel, C=c, gamma=gamma)
#     modelSVM.fit(X, Y)
#     with open("model", "wb") as r:
#         pickle.dump(modelSVM, r)


def classify(df, kernel, c, gamma):
    Y = df.Kelas
    X = vectorizer_tfidf.fit_transform(df['Lowers'])
    model = svm.SVC(kernel=kernel, C=c, gamma=gamma)
    modelSVM = svm.SVC(kernel=kernel, C=c, gamma=gamma)
    modelSVM.fit(X, Y)
    with open("model", "wb") as r:
        pickle.dump(modelSVM, r)
    scores = []
    confusion_score = []
    scores.append(['Uji ke', 'akurasi', 'precision',
                  'recall', 'f- measure', 'waktu Komputasi'])
    cv = KFold(n_splits=10)
    index_hasil = 1
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
        execution_time = round((time.time() - start_time), 4)
        scores.append([index_hasil, acc, pre, rec, f1, execution_time])
        confusion_score.append([tn, fp, fn, tp])
        index_hasil += 1

    temp = ['Rata-rata', 0, 0, 0, 0, 0]
    for i in range(1, 11):
        for j in range(1, 6):
            temp[j] += scores[i][j]

    for i in range(1, 6):
        temp[i] = round((temp[i]/10), 4)

    scores.append(temp)
    scores.append(confusion_score)
    return scores


app = Flask(__name__)

# tfidf = pickle.load(open("tfidf", 'rb'))

with open("model", "rb") as r:
    model = pickle.load(r)

# CORS(app)
CORS(app)


@app.route("/upload", methods=["GET", "POST"])
def upload():

    # response.headers.add('Access-Control-Allow-Origin', '*')
    if request.method == 'POST':
        # try:
        app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, '')
        file = request.files['file']
        print(file)
        filename = secure_filename('dataset.csv')
        # filecsv = filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return ('sukses')
        # except:
        #     return ('data anda salah')

    else:
        return "<h1>Welcome to OOD API BEST PARAMS</h1>"


@app.route("/prepocessing", methods=["GET", "POST"])
def prepocessing():
    if request.method == 'POST':
        try:
            # path = (r"coba\")
            mentah = pd.read_csv("dataset.csv", sep=';')
            mentah = mentah.sample(
                frac=1.0, random_state=1).reset_index(drop=True)
            lower = mentah.astype(str).apply(lambda x: x.str.lower())
            df = lower
            df['Kalimat'] = df['Kalimat'].fillna('').astype(str).str.replace(
                r'[^A-Za-z ]', '', regex=True).replace('', np.nan, regex=False)
            df['Kalimat'] = df.apply(
                lambda row: nltk.word_tokenize(row['Kalimat']), axis=1)
            df['Kalimat'] = df['Kalimat'].apply(stopwords_removal)

            for document in df['Kalimat']:
                for term in document:
                    if term not in term_dict:
                        term_dict[term] = ' '
            for term in term_dict:
                term_dict[term] = stemmed_wrapper(term)

            df['Kalimat'] = df['Kalimat'].swifter.apply(get_stemmed_term)

            df['Propocessing'] = [", ".join(review)
                                  for review in df['Kalimat'].values]
            df['Lowers'] = [", ".join(review)
                            for review in df['Kalimat'].values]
            df.to_csv("Text_prepocessing.csv", sep=";")
            mentah = np.array(mentah).tolist()
            preprocesiing = np.array(df).tolist()

            return ({
                'data': [mentah[0][0], mentah[1][0], preprocesiing[0][2], preprocesiing[1][2]]
            })
        except:
            return ("dataset salah")

    else:
        return "<h1>Welcome to OOD API PREPOCESSING DATASET</h1>"


@app.route("/tfidf", methods=["GET", "POST"])
def download():
    if request.method == 'POST':
        tfidf()
        return('done')
    else:
        app.config['DOWNLOAD_FOLDER'] = os.path.join(app.root_path, '')
        print(app.config['DOWNLOAD_FOLDER'])
        return send_from_directory(app.config['DOWNLOAD_FOLDER'], path='TFIDF.csv', as_attachment=True)


@app.route("/best", methods=["GET", "POST"])
def params():
    if request.method == 'POST':
        try:
            df = pd.read_csv("Text_prepocessing.csv", sep=';')
            params = best_params(df)
            return ({
                'kernel': params['kernel'],
                'c': params['C'],
                'gamma': params['gamma']

            })
        except:
            return ('data anda salah')

    else:
        return "<h1>Welcome to OOD API BEST PARAMS</h1>"


@app.route("/training", methods=["GET", "POST"])
def training():
    if request.method == 'POST':
        # try:
        df = pd.read_csv("Text_prepocessing.csv", sep=';')
        params = request.get_json()
        c = params['c']
        gamma = params['gamma']
        kernel = params['kernel']
        # model(df, kernel, c, gamma)
        score = classify(df, kernel, c, gamma)
        print(c)
        lists = np.array(score[12]).tolist()
        return ({
            'title': score[0],
            'score': score[1:11],
            'means': score[11],
            'confusion': lists
        })
        # except:
        #     return ('Training Gagal')

    else:
        return "<h1>Welcome to OOD API TRINING DATASET</h1>"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        try:
            quest = request.get_json()
            string = quest['text']
            string = string.lower()
            Input = clean_punct(string)
            Input = nltk.word_tokenize(Input)
            Input = stopwords_removal(Input)
            Input = (' ').join(Input)
            Input = stemmed_wrapper(Input)
            Input = vectorizer_tfidf.transform([Input])
            prediction = model.predict(Input)
            return ({
                'pertanyaan': quest['text'],
                'predict': prediction[0]
            })
        except:
            return ('Tidak ada inputan/TFIDF belum tersedia')

    else:
        return "<h1>Welcome to OOD API</h1>"


# if __name__ == "__main__":
#     app.run()
