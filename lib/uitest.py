import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import math
from numpy import dot
from numpy.linalg import norm
from flask import Flask, render_template, request

nltk.download('stopwords')

class CTextSearch:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.MovieData = pd.read_csv("E:/Data Mining/Dataset/craigslist-carstrucks-data/craigslistVehicles.csv")
        self.postings = defaultdict(dict)
        self.docF = defaultdict(int)
        self.dict = set()
        self.length = []
        self.totalDocument = 0
        self.similarity_vec=[]
        self.sorted_similarity = []

    # create token from the string description and remove stop word(common word)
    def tokenize(self,description):
        if pd.isnull(description):
            return []
        else:
            terms = description.lower().split()
            #remove stop word
            filtered = [word for word in terms if not word in stopwords.words('english')]
            return filtered

    #shpuld be called and initialize when server start
    def Read_and_initialise_document(self):
        [self.totalDocument,TotalDimension]  = self.MovieData.shape
        self.totalDocument = 200 #need to comment only for debuging
        for index in range(self.totalDocument):
            terms = self.tokenize(self.MovieData.loc[index, 'desc'])
            self.length.append(len(terms))
            #print(self.length[index])
            # updating dictionary with all available terms
            unique_terms = set(terms)
            self.dict = self.dict.union(unique_terms)
            for term in unique_terms:
                #updating count of each term in posting(document)

                self.postings[term][index] = terms.count(term)
            # self.logObj.progress_track(index,self.totalDocument)

    # should be called and initialize when server start
    def Calculating_Document_frequency(self):
        #calculating and storing for each terms appeared in how many document
        total_term = len(self.dict)
        index = 0
        for term in self.dict:
            self.docF[term] = len(self.postings[term])
            index += 1
            # self.logObj.progress_track(index, total_term)


    def Search(self,query):
        #check if query is empty or query is not available in any of document
        self.similarity_vec = []
        query_token = self.tokenize(query)
        flag = False
        if query_token == []:
            flag = False
        else:
            for term in query_token:
                if term in self.dict:
                    flag = True
        if flag == False:
            return []

        Query_vector = self.Make_Query_vector(query_token)

        for index in range(self.totalDocument):
            document_Vector = self.Make_Document_vector(query_token,index)
            similarity = self.Calculate_similarity(Query_vector,document_Vector)
            self.similarity_vec.append(similarity)
        self.sorted_similarity = np.argsort(self.similarity_vec)
        #print(self.sorted_similarity)

        movie_name = []
        movie_description = []
        movie_price=[]
        retrive_data = {}
        for result_index in range(5):
            movie_name.append(self.MovieData.loc[self.sorted_similarity[result_index],'manufacturer'])
            movie_price.append(self.MovieData.loc[self.sorted_similarity[result_index],'price'])
            movie_description.append(self.MovieData.loc[self.sorted_similarity[result_index], 'desc'])
        retrive_data.update({"Movie":movie_name})
        retrive_data.update({"Description": movie_description})
        retrive_data.update({"Price": movie_price})
        print(retrive_data)
        return  retrive_data

    #calculate cosign similarity of two tf-idf vector
    def Calculate_similarity(self,query_vec,doc_vec):

        return dot(query_vec,doc_vec)/(norm(query_vec)*norm(doc_vec))

    # create tf-idf weight vector of query term
    def Make_Query_vector(self,query_token):
        unique_Q_terms = set(query_token)
        query_length = len(query_token)
        vector = []
        for term in unique_Q_terms:
            term_count = query_token.count(term)
            term_F = term_count / query_length
            query_idf = self.Calculate_Inverse_Document_Frequency(term)
            tf_idf = term_F * query_idf
            vector.append(tf_idf)

        return vector

    # create tf-idf weight vector of query term for document
    def Make_Document_vector(self,query_token,id):
        unique_Q_terms = set(query_token)
        document_vector = []
        for q_term in unique_Q_terms:
            if q_term in self.dict:
                if id in self.postings[q_term]:
                    tf = self.postings[q_term][id] / self.length[id]
                    tf_idf = tf * self.Calculate_Inverse_Document_Frequency(q_term)
                    document_vector.append(tf_idf)
                else:
                    document_vector.append(0)
            else:
                document_vector.append(0)
        #print("Document id:")
        #print(id)
        #print(document_vector)
        return document_vector

    def Calculate_Inverse_Document_Frequency(self,term):
        #calculate idf for given term
        #idf = log(Number of document / Number of document term appeared)
        if term in self.dict:
            return math.log(self.totalDocument/self.docF[term],2)
        else:
            return 0

    def getFileReadObj(self):
        return self.FileHandlingObj

    #added for debuging purpose
    def DisplayData(self):
        #print(self.MovieData.loc[:,'overview'])
        #print(self.MovieData.head())
        print(stopwords.words('english'))

        #print(self.MovieData('titile'))
        #print(self.MovieData.shape)

# app = Flask(__name__)
#
#
# @app.route('/')
# def homepage():
#     title = "Epic Tutorials"
#     paragraph = [
#         "wow I am learning so much great stuff!wow I am learning so much great stuff!wow I am learning so much great stuff!wow I am learning so much great stuff!",
#         "wow I am learning so much great stuff!wow I am learning so much great stuff!wow I am learning so much great stuff!wow I am learning so much great stuff!wow I am learning so much great stuff!wow I am learning so much great stuff!wow I am learning so much great stuff!wow I am learning so much great stuff!wow I am learning so much great stuff!"]
#
#     try:
#         return render_template("index.html", title=title, paragraph=paragraph)
#     except Exception, e:
#         return str(e)
#
#
# @app.route('/results', methods=['POST'])
# def resultspage():
#     title = "About this site"
#     paragraph = ["blah blah blah memememememmeme blah blah memememe"]
#     CTextSearch.Search(self, request.form['textinput'])
#     pageType = 'about'
#     return render_template("index.html", title=title, paragraph=paragraph, pageType=pageType)
#
#
# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=8080, passthrough_errors=True)

