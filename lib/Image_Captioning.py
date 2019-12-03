import numpy as np
import pandas as pd
import math as mt
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer, LancasterStemmer
from nltk.corpus import stopwords
import nltk
from collections import defaultdict
import math
from numpy import dot
from numpy.linalg import norm
from collections import OrderedDict
import time
import logging

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

logger = logging.getLogger()


class ImageCaptionPhase:
    def __init__(self):
        # self.UsedCarsDS = pd.read_csv(
        #     "E:/Data Mining/Dataset/smaller dataset/craigslistVehicles/3.7 dataset/craigslistVehiclesCheck.csv")
        self.UsedCarsDS = pd.read_csv(
            "/home/recklessPaul94/craigslistVehiclesCheck.csv")
        self.inverted_index = defaultdict(dict)
        self.wordFreqInDocs = defaultdict(int)
        self.uniqueWordsSet = set()
        self.lengthOfDocuments = []
        self.totalRows = 0
        self.ranked_rows = []
        self.calculations_dict = {}
        self.cosine_idx_calculations = {}

    # we tokenize to remove all the stopwords and return a set of words
    def tokenize(self, description):
        filtered = []
        if pd.isnull(description):
            return []
        else:
            terms = description.lower().split()
            # terms = word_tokenize(description.lower().decode('utf-8'))
            filtered_stopwords = [word for word in terms if not word in stopwords.words('english')]

            # # Stemming Snowball
            # stemmer = SnowballStemmer('english')
            # for stem in filtered_stopwords:
            #     filtered.append(stemmer.stem(stem.decode('utf-8')))

            # # Stemming Porter
            # stemmer = PorterStemmer()
            # for stem in filtered_stopwords:
            #     filtered.append(stemmer.stem(stem.decode('utf-8')))

            # Lemmatizer Word Net Lemmatizer
            lemmatizer = WordNetLemmatizer()
            for lemmatized in filtered_stopwords:
                filtered.append(lemmatizer.lemmatize(lemmatized))

            filtered_final = []
            # Stemming Lancaster
            stemmer = LancasterStemmer()
            for stem in filtered:
                # filtered_final.append(stemmer.stem(stem.decode('utf-8')))
                filtered_final.append(stemmer.stem(stem))

            # # Lemmatizer TextBlob
            # for lemmatized in filtered_stopwords:
            #     w = Word(lemmatized.decode('utf-8'))
            #     filtered.append(w.lemmatize)

            return filtered_stopwords, filtered_final

    # i am doing this to load the dataset on the server and calculate for the words that are there on the server
    # so i don't have to compute everytime the query comes
    def create_inverted_index(self):
        # i am doing this so i get the total number of documents
        try:
            logger.info("ImageCaption: Creating Image Caption Inverted Index")
            self.totalRows = len(self.UsedCarsDS)
            self.totalRows = 200
            for idx in self.UsedCarsDS.index:
                if idx == self.totalRows:
                    break
                filtered, words = self.tokenize(self.UsedCarsDS.get_value(idx, 'captions'))
                self.lengthOfDocuments.append(len(words))
                # i am using 'set' to remove all the repeated words from the tokenized description
                unique_terms = set(words)
                # adding all unique terms in the uniqueWordsSet
                self.uniqueWordsSet = self.uniqueWordsSet.union(unique_terms)
                for word in unique_terms:
                    # i am using index coz every document i find this same word
                    # it will add the index number and the value instead of overwriting it
                    self.inverted_index[word][idx] = words.count(word)
            logger.info("ImageCaption: Inverted index Loop ran for {} rows".format(idx))
        except Exception as e:
            logger.error("ImageCaption: Exception while creating index : {}".format(e))

    # Calculating the document frequency
    def document_frequency(self):
        # here i am getting the length of my inverted index value for each WORD
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
        car_caption = []
        car_price = []
        calculations = []
        cosine = []
        image_url = []
        results_payload = OrderedDict()
        self.calculations_dict = {}

        filtered, input_string_tokenized = self.tokenize(input_string)
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

        if not flag:
            return []

        input_vector = self.create_input_string_vector(input_string_tokenized)
        for index in range(self.totalRows):
            row_vector = self.create_row_vector(input_string_tokenized, index)
            similarity = self.cosine_similarity(input_vector, row_vector)
            if math.isnan(similarity):
                similarity = 0.0
            self.cosine_idx_calculations[index] = similarity
            similarity_list.append(similarity)
        # numpy's argsort will sort the list based on similarity
        # and return its indexes which i will use later for retrieving
        self.ranked_rows = np.argsort(similarity_list)

        for result_index in range(5):
            # if we dont add the name of the column at the end it will retrieve the whole row
            manufacturer_name.append(
                self.UsedCarsDS.loc[self.ranked_rows[(len(self.ranked_rows) - 1) - result_index], 'manufacturer'])
            car_price.append(self.UsedCarsDS.loc[self.ranked_rows[(len(self.ranked_rows) - 1) - result_index], 'price'])
            # test_desc = str(
            #     self.UsedCarsDS.loc[self.ranked_rows[(len(self.ranked_rows) - 1) - result_index], 'desc']).decode(
            #     "utf-8")
            test_desc = str(
                self.UsedCarsDS.loc[self.ranked_rows[(len(self.ranked_rows) - 1) - result_index], 'captions'])
            for qwe in filtered:
                test_desc = test_desc.lower().replace(" " + qwe + " ",
                                                      "<span style='color:#FF0000'> " + qwe + " </span>")
            car_caption.append(test_desc)
            calculations.append(
                self.calculations_dict.get(self.ranked_rows[(len(self.ranked_rows) - 1) - result_index]))
            cosine.append(
                self.cosine_idx_calculations.get(self.ranked_rows[(len(self.ranked_rows) - 1) - result_index]))
            url = self.UsedCarsDS.loc[self.ranked_rows[(len(self.ranked_rows) - 1) - result_index], 'image_url']
            if not url:
                image_url.append("")
            else:
                image_url.append(
                    self.UsedCarsDS.loc[self.ranked_rows[(len(self.ranked_rows) - 1) - result_index], 'image_url'])

        for test in range(5):
            testdict = OrderedDict()
            testdict["Manufacturer"] = str(manufacturer_name[test])
            testdict["Price"] = str(car_price[test])
            testdict["Caption"] = car_caption[test]
            testdict["Calculation"] = calculations[test]
            testdict["Cosine"] = cosine[test]
            testdict["Image"] = image_url[test]
            results_payload[str(test)] = testdict

        logging.debug("Image Caption payload")
        logging.debug(results_payload)
        return results_payload

    def search_dataset_with_image(self, input_string):
        similarity_list = []
        manufacturer_name = []
        car_caption = []
        car_price = []
        calculations = []
        cosine = []
        image_url = []
        results_payload = OrderedDict()
        self.calculations_dict = {}

        filtered, input_string_tokenized = self.tokenize(input_string)
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

        if not flag:
            return []

        input_vector = self.create_input_string_vector(input_string_tokenized)
        for index in range(self.totalRows):
            row_vector = self.create_row_vector(input_string_tokenized, index)
            similarity = self.cosine_similarity(input_vector, row_vector)
            if math.isnan(similarity):
                similarity = 0.0
            self.cosine_idx_calculations[index] = similarity
            similarity_list.append(similarity)
        # numpy's argsort will sort the list based on similarity
        # and return its indexes which i will use later for retrieving
        self.ranked_rows = np.argsort(similarity_list)

        for result_index in range(5):
            # if we dont add the name of the column at the end it will retrieve the whole row
            manufacturer_name.append(
                self.UsedCarsDS.loc[self.ranked_rows[(len(self.ranked_rows) - 1) - result_index], 'manufacturer'])
            car_price.append(self.UsedCarsDS.loc[self.ranked_rows[(len(self.ranked_rows) - 1) - result_index], 'price'])
            # test_desc = str(
            #     self.UsedCarsDS.loc[self.ranked_rows[(len(self.ranked_rows) - 1) - result_index], 'desc']).decode(
            #     "utf-8")
            test_desc = str(
                self.UsedCarsDS.loc[self.ranked_rows[(len(self.ranked_rows) - 1) - result_index], 'captions'])
            for qwe in filtered:
                test_desc = test_desc.lower().replace(" " + qwe + " ",
                                                      "<span style='color:#FF0000'> " + qwe + " </span>")
            car_caption.append(test_desc)
            calculations.append(
                self.calculations_dict.get(self.ranked_rows[(len(self.ranked_rows) - 1) - result_index]))
            cosine.append(
                self.cosine_idx_calculations.get(self.ranked_rows[(len(self.ranked_rows) - 1) - result_index]))
            url = self.UsedCarsDS.loc[self.ranked_rows[(len(self.ranked_rows) - 1) - result_index], 'image_url']
            if not url:
                image_url.append("")
            else:
                image_url.append(
                    self.UsedCarsDS.loc[self.ranked_rows[(len(self.ranked_rows) - 1) - result_index], 'image_url'])

        for test in range(5):
            testdict = OrderedDict()
            testdict["Manufacturer"] = str(manufacturer_name[test])
            testdict["Price"] = str(car_price[test])
            testdict["Caption"] = car_caption[test]
            testdict["Calculation"] = calculations[test]
            testdict["Cosine"] = cosine[test]
            testdict["Image"] = image_url[test]
            results_payload[str(test)] = testdict

        logging.debug("Image Caption payload")
        logging.debug(results_payload)
        return results_payload

    # cosine similarity = (A . B) / ||A|| ||B|| --- I am using numpy to get those values and then multiply/divide them
    def cosine_similarity(self, query_vec, doc_vec):
        return dot(query_vec, doc_vec) / (norm(query_vec) * norm(doc_vec))

    # creation of vector for input
    def create_input_string_vector(self, input_string_tokenized):
        unique_words = sorted(set(input_string_tokenized), key=input_string_tokenized.index)
        input_string_length = len(input_string_tokenized)
        vector = []
        for word in unique_words:
            word_count = input_string_tokenized.count(word)
            term_freq = float(word_count) / input_string_length
            term_idf = self.calculate_idf(word)
            tf_idf = term_freq * term_idf
            vector.append(tf_idf)

        return vector

    # creation of row vector to calculate similarity later
    def create_row_vector(self, input_string_tokenized, idx):
        input_unique_words = sorted(set(input_string_tokenized), key=input_string_tokenized.index)
        row_vector = []
        word_dict = {}
        for input_word in input_unique_words:
            if input_word in self.uniqueWordsSet:
                if idx in self.inverted_index[input_word]:
                    calc_dict = {}
                    term_freq = float(self.inverted_index[input_word][idx]) / self.lengthOfDocuments[idx]
                    calc_dict['tf'] = term_freq
                    idf = self.calculate_idf(input_word)
                    calc_dict['idf'] = idf
                    tf_idf = term_freq * idf
                    calc_dict['tf-idf'] = tf_idf
                    row_vector.append(tf_idf)
                    word_dict[input_word] = calc_dict
                else:
                    row_vector.append(0)
            else:
                row_vector.append(0)
        self.calculations_dict[idx] = word_dict
        return row_vector

    def calculate_idf(self, word):
        # wordFreqInDocs will return the integer value of
        # how many documents in the data set has the current word occurred
        if word in self.uniqueWordsSet:
            return math.log(float(self.totalRows) / self.wordFreqInDocs[word], 2)
        else:
            return 0


# import pandas as pd
#
# # Write a code to read data set and paste values
# dataset = pd.read_csv("drive/My Drive/craigslistVehicles.csv")
#
# test = []
# for idx in range(len(dataset)):
#     image_url = dataset['image_url'][idx]
#     image_extension = image_url[-4:]
#     image_path = tf.keras.utils.get_file('image' + image_extension,
#                                          origin=image_url)
#
#     result, attention_plot = evaluate(image_path)
#     test.append(' '.join(result))
#     # print('Prediction Caption:', ' '.join(result))
#     # plot_attention(image_path, result, attention_plot)
#     # opening the image
#
# dataset.insert(22, 'Captions', test)
# print("Hello")
#
# dataset.to_csv(r'drive/My Drive/craigslistVehiclesTest.csv')


