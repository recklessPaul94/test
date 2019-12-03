from math import log
import pandas as pd
from flask import Flask, render_template, request, jsonify
# import TfIDFSearch as ut
# import Classifier as cf
import TfIDFSearch as ut
import Classifier as cf
import Image_Captioning as ic
# import ImageEvaluator as iv
from flask_bootstrap import Bootstrap
from collections import OrderedDict
import logging
# import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

app = Flask(__name__)
Bootstrap(app)

SearchObj = ut.SearchPhase()
ClassifierObj = cf.Naive_bayes_classifier()
# SearchObj = ""
# ClassifierObj = ""
ImageObj = ic.ImageCaptionPhase()
# iv.prepare()

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
        return render_template("imagecaptionpage.html", data=testdict, imageObj=testdict)
    except Exception as e:
        return str(e)


@app.route('/imagecaptionresults', methods=['GET', 'POST'])
def imagecaptionresultspage():
    testdict = OrderedDict()
    inputquery = (request.form['textinput'])
    Results = ImageObj.search_dataset(inputquery)
    pageType = 'about'
    return render_template("imagecaptionpage.html", data=eval(str(Results)), imageObj=testdict)


IMAGES_PATH = '/home/recklessPaul94/images/'
# IMAGES_PATH = 'E:/Data Mining/imagesTest/'

# @app.route('/imagecaptionresultsimage', methods=['GET', 'POST'])
# def imagecaptionresults():
#     f = request.files['img_file']
#     image_extension = f.filename[-4:]
#     f.save(IMAGES_PATH +"Ashley"+str(image_extension))
#
#     image_url = IMAGES_PATH +(f.filename)
#     if not image_url:
#         return render_template("imagecaptionpage.html", data="")
#
#     if type(image_url) != str:
#         return render_template("imagecaptionpage.html", data="")
#
#     image_url_final = str(image_url).strip()
#     if len(image_url_final) < 5:
#         return render_template("imagecaptionpage.html", data="")
#
#     # image_extension = image_url_final[-4:]
#     # if "jpg" not in image_extension:
#     #     return render_template("imagecaptionpage.html", data="")
#     try:
#         caption_string = ""
#         # image_path = tf.keras.utils.get_file(('image7') + image_extension,
#         #                                      origin=image_url_final)
#         result, attention_plot = iv.evaluate(image_url)
#         caption_string = ' '.join(result)
#         Results = ImageObj.search_dataset(caption_string)
#         inputImgMap = {}
#         inputImgMap['image_url'] = image_url
#         inputImgMap['input_caption'] = caption_string
#
#     except Exception as e:
#         print(e)
#
#     return render_template("imagecaptionpage.html", data=eval(str(Results)), imageObj=eval(str(inputImgMap)))



logger.error("Initializing Search Object")
InitialiseSearchObject()
InitializeClassifierObject()
IntializeImageCaptionObject()

print("Hello Priyana, you're awesome")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080, passthrough_errors=True,use_reloader=False)
