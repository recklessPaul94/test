from math import log
import pandas as pd
from flask import Flask, render_template, request, jsonify
import TfIDFSearch as ut
from flask_bootstrap import Bootstrap
from collections import OrderedDict
import json

ClassObj = ut.SearchPhase()

# inverseDict = {}
#
# pd.set_option('display.max_columns', None)
# myFile = pd.read_csv("E:/Data Mining/Dataset/craigslist-carstrucks-data/craigslistVehicles.csv")
#
# print (myFile.head(0))
# print (myFile.head(1))
#
#
# def termFreq(word, docDescription):
#     normalizeDocDescription = docDescription.lower().split()
#     docTerms = normalizeDocDescription.count(word.lower())
#     doc_length = float(len(normalizeDocDescription))
#     normalizedTermFreq = docTerms / doc_length
#     return normalizedTermFreq
#
#
# def inverseDocFreq(word):
#     docTermsNum = 0
#     totalDocs = 0
#     if word in inverseDict:
#         return inverseDict[word]
#     try:
#         for index, line1 in myFile.iterrows():
#             docDescription1 = str(line1['desc'])
#             if word.lower() in docDescription1.lower():
#                 docTermsNum += 1
#         if docTermsNum > 0:
#             idf_value = log(float(myFile.size / docTermsNum))
#             inverseDict[word] = idf_value
#             return idf_value
#         else:
#             return 0
#     except Exception as e:
#         print(e)
#         return 0
#
# def myBaseFunction(inputstring):
#     for index, line in myFile.iterrows():
#         rank = 0
#         for string in str[inputstring]:
#             try:
#                 docDescription = line['desc']
#                 termfrequency = termFreq(string, docDescription)
#                 idf_value = inverseDocFreq(string)
#                 rank += termfrequency * idf_value
#                 print(rank)
#             except Exception as e:
#                 print(e)
#                 continue

def InitialiseSearchObject():
    ClassObj.create_inverted_index()
    ClassObj.document_frequency()


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


@app.route('/results', methods=['POST'])
def resultspage():
    inputquery = (request.form['textinput'])
    Results = ClassObj.search_dataset(inputquery)
    pageType = 'about'
    return render_template("homepage.html", data=eval(str(Results)))
    # return render_template("homepage.html")


InitialiseSearchObject()

print("Hello Priyana, you're awesome")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080, passthrough_errors=True)