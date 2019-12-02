from math import log
import pandas as pd
from flask import Flask, render_template, request, jsonify
# import TfIDFSearch as ut
# import Classifier as cf
import TfIDFSearch as ut
import Classifier as cf
import Image_Captioning as ic
from flask_bootstrap import Bootstrap
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

app = Flask(__name__)
Bootstrap(app)

SearchObj = ut.SearchPhase()
ClassifierObj = cf.Naive_bayes_classifier()
ImageObj = ic.ImageCaptionPhase()

def InitialiseSearchObject():
    SearchObj.create_inverted_index()
    SearchObj.document_frequency()


def InitializeClassifierObject():
    ClassifierObj.initialize_class_wise_inverted_index()


def IntializeImageCaptionObject():
    ImageObj.create_inverted_index()
    ImageObj.document_frequency()

@app.before_first_request
def logging_init():
    logging.basicConfig()

# Search
@app.route('/', methods=['GET'])
def homepage():
    try:
        testdict = OrderedDict()
        return render_template("homepage.html", data=testdict)
    except Exception as e:
        print(e)
        return str(e)


@app.route('/results', methods=['POST'])
def resultspage():
    inputquery = (request.form['textinput'])
    Results = SearchObj.search_dataset(inputquery)
    pageType = 'about'
    return render_template("homepage.html", data=eval(str(Results)))

# Classifier
@app.route('/classifier', methods=['GET'])
def classifierpagecontroller():
    try:
        testdict = OrderedDict()
        return render_template("classifierpage.html", data=testdict)
    except Exception as e:
        print(e)
        return str(e)


@app.route('/classifierresults', methods=['POST'])
def classifierresultspage():
    inputquery = (request.form['descinput'])
    Results = ClassifierObj.classify_dataset(inputquery, SearchObj)
    return render_template("classifierresultspage.html", data=eval(str(Results)))

# Image Captioning
@app.route('/imagecaption', methods=['GET'])
def imagecaptionpage():
    try:
        testdict = OrderedDict()
        return render_template("imagecaptionpage.html", data=testdict)
    except Exception as e:
        return str(e)


@app.route('/imagecaptionresults', methods=['POST'])
def imagecaptionresultspage():
    inputquery = (request.form['textinput'])
    Results = ImageObj.search_dataset(inputquery)
    pageType = 'about'
    return render_template("imagecaptionpage.html", data=eval(str(Results)))


logger.error("Initializing Search Object")
InitialiseSearchObject()
InitializeClassifierObject()
IntializeImageCaptionObject()

print("Hello Priyana, you're awesome")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080, passthrough_errors=True,use_reloader=False)
