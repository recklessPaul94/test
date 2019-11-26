from math import log
import pandas as pd
from flask import Flask, render_template, request, jsonify
import TfIDFSearch as ut
import Classifier as cf
from flask_bootstrap import Bootstrap
from collections import OrderedDict

SearchObj = ut.SearchPhase()
ClassifierObj = cf.Naive_bayes_classifier()


def InitialiseSearchObject():
    SearchObj.create_inverted_index()
    SearchObj.document_frequency()


def InitializeClassifierObject():
    ClassifierObj.initialize_class_wise_inverted_index()


app = Flask(__name__)
Bootstrap(app)


@app.route('/', methods=['GET'])
def homepage():
    try:
        testdict = OrderedDict()
        return render_template("homepage.html", data=testdict)
    except Exception as e:
        print(e)
        return str(e)

@app.route('/classifier', methods=['GET'])
def classifierpagecontroller():
    try:
        testdict = OrderedDict()
        return render_template("classifierpage.html", data=testdict)
    except Exception as e:
        print(e)
        return str(e)

@app.route('/results', methods=['POST'])
def resultspage():
    inputquery = (request.form['textinput'])
    Results = SearchObj.search_dataset(inputquery)
    pageType = 'about'
    return render_template("homepage.html", data=eval(str(Results)))
    # return render_template("homepage.html")

@app.route('/classifierresults', methods=['POST'])
def classifierresultspage():
    inputquery = (request.form['descinput'])
    Results = ClassifierObj.classify_dataset(inputquery, SearchObj)
    return render_template("classifierresultspage.html", data=eval(str(Results)))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080, passthrough_errors=True)

InitialiseSearchObject()
InitializeClassifierObject()

print("Hello Priyana, you're awesome")