from process_preprocessing import preprocessing
from process_tfidf import tfidf
from process_best import best_params
from process_train import training
from process_detect import detect
from werkzeug.utils import secure_filename
from flask import Flask, request, send_from_directory
import numpy as np
import os
from flask_cors import CORS
from flask import send_from_directory


app = Flask(__name__)
CORS(app)


class main():
    @app.route("/upload", methods=["GET", "POST"])
    def save_file():
        if request.method == 'POST':
            try:
                app.config['UPLOAD_FOLDER'] = os.path.join(
                    app.root_path, 'data')
                file = request.files['file']
                print(file)
                filename = secure_filename('dataset.csv')
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return ('succses')
            except:
                return ("0")

        else:
            return "<h1>Welcome to OOD API BEST PARAMS</h1>"

    @app.route("/preprocessing", methods=["GET", "POST"])
    def preprocessing_api():
        if request.method == 'POST':
            try:
                preprocess = preprocessing()
                preprocess.read_data()
                preprocess.clean_punct()
                preprocess.lower_case()
                preprocess.stopwords_removal()
                preprocess.stemming()
                row, result = preprocess.save_preprocessing()
                return ({
                    'data': [row[0][0], row[1][0], result[0][0], result[1][0]]
                })
            except:
                return ("0")

        else:
            return "<h1>Welcome to OOD API PREPROCESSING DATASET</h1>"

    @app.route("/tfidf", methods=["GET", "POST"])
    def tfidf_api():
        try:
            if request.method == 'POST':
                result_tfidf = tfidf()
                result_tfidf.read_data()
                result_tfidf.tfidf()
                result_tfidf.save_tfidf()
                return('done')
            else:
                app.config['DOWNLOAD_FOLDER'] = os.path.join(
                    app.root_path, 'data')
                print(app.config['DOWNLOAD_FOLDER'])
                return send_from_directory(app.config['DOWNLOAD_FOLDER'], path='TFIDF.csv', as_attachment=True)
        except:
            return ("0")

    @app.route("/best", methods=["GET", "POST"])
    def best_api():
        if request.method == 'POST':
            try:
                best = best_params()
                best.read_data()
                acc, kernel, c = best.best_params()
                return ({
                    'kernel': kernel,
                    'c': c,
                    'accurasy': acc
                })
            except:
                return ("0")

        else:
            return "<h1>Welcome to OOD API BEST PARAMS</h1>"

    @app.route("/training", methods=["GET", "POST"])
    def training_api():
        if request.method == 'POST':
            try:
                params = request.get_json()
                clasify = training()
                clasify.kernel = params['kernel']
                clasify.c = params['c']
                clasify.read_data()
                score = clasify.classify()
                lists = np.array(score[12]).tolist()
                return ({
                    'score': score[1:12],
                    'means': score[11],
                    'confusion': lists
                })
            except:
                return ("0")

        else:
            return "<h1>Welcome to OOD API TRINING DATASET</h1>"

    @app.route("/", methods=["GET", "POST"])
    def detection_api():
        if request.method == 'POST':
            try:
                quest = request.get_json()
                string = quest['text']
                prediction = detect()
                prediction.text = string
                prediction.clean_punct()
                prediction.lower_case()
                prediction.stopwords_removal()
                prediction.stemming()
                prediction = prediction.predict()
                return ({
                    'text': quest['text'],
                    'predict': prediction[0]
                })
            except:
                return ("0")

        else:
            return "<h1>Welcome to OOD API</h1>"
