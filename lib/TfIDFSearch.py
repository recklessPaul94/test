import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import math
from numpy import dot
from numpy.linalg import norm


class CTextSearch:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.UsedCarsDS = pd.read_csv("E:/Data Mining/Dataset/craigslist-carstrucks-data/craigslistVehicles.csv")
        self.inverted_index = defaultdict(dict)
        self.wordFreqInDocs = defaultdict(int)
        self.uniqueWordsSet = set()
        self.lengthOfDocuments = []
        self.totalRows = 0
        # self.similarity_vec=[]
        self.ranked_rows = []

    # create token from the string description and remove stop word(common word)
    def tokenize(self,description):
        if pd.isnull(description):
            return []
        else:
            terms = description.lower().split()
            #remove stop word
            filtered = [word for word in terms if not word in stopwords.words('english')]
            return filtered

    # i am doing this to load the dataset on the server and calculate for the words that are there on the server
    # so i don't have to compute everytime the query comes
    def Read_and_initialise_document(self):
        # i am doing this so i get the total number of documents
        self.totalRows = len(self.UsedCarsDS)
        self.totalRows = 100
        for idx in range(self.totalRows):
            words = self.tokenize(self.UsedCarsDS.loc[idx, 'desc'])
            self.lengthOfDocuments.append(len(words))
            # i am using 'set' to remove all the repeated words from the tokenized description
            unique_terms = set(words)
            # adding all unique terms in the uniqueWordsSet
            self.uniqueWordsSet = self.uniqueWordsSet.union(unique_terms)
            for word in unique_terms:
                # i am using index coz every document i find this same word
                # it will add the index number and the value instead of overwriting it
                self.inverted_index[word][idx] = words.count(word)

    # should be called and initialize when server start
    def Calculating_Document_frequency(self):
        # here i am getting the length of my dictionary value for each WORD
        # so ill know the number of documents its been in
        for term in self.uniqueWordsSet:
            self.wordFreqInDocs[term] = len(self.inverted_index[term])

    def termFreq(word, docDescription):
        normalizeDocDescription = docDescription.lower().split()
        docTerms = normalizeDocDescription.count(word.lower())
        doc_length = float(len(normalizeDocDescription))
        normalizedTermFreq = docTerms / doc_length
        return normalizedTermFreq


    def search_dataset(self, input_string):
        similarity_list = []
        manufacturer_name = []
        car_description = []
        car_price = []
        results_payload = {}

        input_string_tokenized = self.tokenize(input_string)
        flag = False
        # it will be empty in case the user only put 'stopwords' or never put any input at all
        if not input_string_tokenized:
            flag = False
        else:
            # we check if any of the words in the input are there in the data set because if none of them are present
            # then we don't need to compute coz it we wont get any matches
            for input_word in input_string_tokenized:
                if input_word in self.uniqueWordsSet:
                    flag = True
        if flag == False:
            return []

        input_vector = self.create_input_string_vector(input_string_tokenized)

        for index in range(self.totalRows):
            row_vector = self.create_row_vector(input_string_tokenized, index)
            similarity = self.cosine_similarity(input_vector, row_vector)
            similarity_list.append(similarity)
        # numpy's argsort will sort the list based on similarity
        # and return its indexes which i will use later for retrieving
        self.ranked_rows = np.argsort(similarity_list)

        for result_index in range(10):
            manufacturer_name.append(self.UsedCarsDS.loc[self.ranked_rows[result_index], 'manufacturer'])
            car_price.append(self.UsedCarsDS.loc[self.ranked_rows[result_index], 'price'])
            car_description.append(self.UsedCarsDS.loc[self.ranked_rows[result_index], 'desc'])
        results_payload.update({"Manufacturer":manufacturer_name})
        results_payload.update({"Price": car_price})
        results_payload.update({"Description": car_description})
        print(results_payload)
        return results_payload

    # cosine similarity = (A . B) / ||A|| ||B|| --- I am using numpy to get those values and then multiply/divide them
    def cosine_similarity(self, query_vec, doc_vec):
        return dot(query_vec,doc_vec)/(norm(query_vec)*norm(doc_vec))

    # creation of vector for input
    def create_input_string_vector(self, input_string_tokenized):
        unique_words = set(input_string_tokenized)
        input_string_length = len(input_string_tokenized)
        vector = []
        for word in unique_words:
            word_count = input_string_tokenized.count(word)
            term_freq = word_count / input_string_length
            term_idf = self.calculate_idf(word)
            tf_idf = term_freq * term_idf
            vector.append(tf_idf)

        return vector

    # creation of row vector to calculate similarity later
    def create_row_vector(self, input_string_tokenized, idx):
        input_unique_words = set(input_string_tokenized)
        row_vector = []
        for input_word in input_unique_words:
            if input_word in self.uniqueWordsSet:
                if idx in self.inverted_index[input_word]:
                    tf = self.inverted_index[input_word][idx] / self.lengthOfDocuments[idx]
                    tf_idf = tf * self.calculate_idf(input_word)
                    row_vector.append(tf_idf)
                else:
                    row_vector.append(0)
            else:
                row_vector.append(0)
        return row_vector

    def calculate_idf(self, word):
        # wordFreqInDocs will return the integer value of
        # how many documents in the data set has the current word occurred
        if word in self.uniqueWordsSet:
            return math.log(self.totalRows / self.wordFreqInDocs[word], 2)
        else:
            return 0


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

