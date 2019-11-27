from nltk import WordNetLemmatizer, LancasterStemmer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
from collections import defaultdict
from collections import Counter
from operator import itemgetter
import TfIDFSearch as ut
import operator
import math
from collections import OrderedDict

# SearchObj = ut.SearchPhase()

class Naive_bayes_classifier():
    def __init__(self):
        # self.UsedCarsDS = pd.read_csv(
        #     "E:/Data Mining/Dataset/smaller dataset/craigslistVehicles/craigslistVehicles.csv")
        self.UsedCarsDS = pd.read_csv(
            "/home/recklessPaul94/craigslistVehiclesCheck.csv")
        self.total_rows = 0
        self.class_wise_dict = {}
        self.inverted_index = {}
        self.unique_word_set_global = set()
        self.labels = []
        self.label_size_count = {}

    def initialize_class_wise_inverted_index(self):
        try:
            self.total_rows = len(self.UsedCarsDS)
            self.total_rows = 2000
            for idx in range(self.total_rows):
                words = self.UsedCarsDS.loc[idx, 'desc']
                unique_words = (self.tokenize(words))
                unique_words_set = set(unique_words)
                self.unique_word_set_global = self.unique_word_set_global.union(unique_words_set)
                label = self.UsedCarsDS.loc[idx, 'type']

                if not label:
                    continue

                if 'nan' in str(label):
                    continue

                if label in self.label_size_count:
                    self.label_size_count[label] = self.label_size_count[label] + 1
                else:
                    self.label_size_count[label] = 1

                for word in unique_words_set:
                    if label in self.class_wise_dict:
                        temp_dict = {}
                        temp_dict = self.class_wise_dict[label]
                        if word not in temp_dict:
                            # temp_dict[word] = unique_words.count(word)
                            temp_dict[word] = 1
                        else:
                            # temp_dict[word] = temp_dict[word] + unique_words.count(word)
                            temp_dict[word] = temp_dict[word] + 1
                        self.class_wise_dict[label] = temp_dict
                    else:
                        # temp_dict = {word: unique_words.count(word)}
                        temp_dict = {word: 1}
                        self.class_wise_dict[label] = temp_dict
                        self.labels.append(label)
        except Exception as e:
            print(e)

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
                filtered.append(lemmatizer.lemmatize(lemmatized.decode('utf-8')))

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

            return filtered_final

    def classify_dataset(self, input_string, SearchObj):
        input_string_tokenized = self.tokenize(input_string)
        validate = False

        if not input_string_tokenized:
            validate = False
        else:
            for each_word in input_string_tokenized:
                if each_word in self.unique_word_set_global:
                    validate = True
                    break

        if not validate:
            return []

        final_dict = {}
        sum_all_label_probability = 0
        # Now we need to classify for every type of car
        for label in self.labels:
            # Smoothing
            current_label_size = self.label_size_count[label] + 1
            current_label_dict = self.class_wise_dict[label]

            probability_all_evidence = 1

            for input_word in input_string_tokenized:
                if input_word not in current_label_dict:
                    count_of_word = 1
                else:
                    count_of_word = current_label_dict[input_word] + 1
                probability_each_evidence = float(count_of_word) / current_label_size
                probability_all_evidence *= probability_each_evidence

            probability_hypothesis = float(current_label_size) / self.total_rows + 1
            final_label_probability = float(probability_all_evidence * probability_hypothesis)

            final_dict[label] = final_label_probability
            sum_all_label_probability += final_label_probability

        sorted_x = sorted(final_dict.items(), key=operator.itemgetter(1), reverse=True)

        final_label_ranked = []
        final_label_percentage_ranked = []
        for tup in sorted_x:
            final_label_ranked.append(tup[0])
            final_label_percentage_ranked.append(round((tup[1]/sum_all_label_probability)*100, 2))

        search_results = SearchObj.search_dataset_with_label(input_string, final_label_ranked[0])

        results_payload = [final_label_ranked, final_label_percentage_ranked, search_results]

        # manufacturer_name = []
        # car_price = []
        # car_description = []
        # label_no = 0
        # for row in range(self.total_rows):
        #     classified_label = final_label_ranked[0]
        #     current_label = self.UsedCarsDS.loc[row, 'type']
        #     if not current_label:
        #         continue
        #     if classified_label is current_label:
        #         manufacturer_name.append(self.UsedCarsDS.loc[row, 'manufacturer'])
        #         car_price.append(self.UsedCarsDS.loc[row, 'price'])
        #         car_description.append(str(self.UsedCarsDS.loc[row, 'desc']).decode("utf-8"))
        #         label_no += 1
        #
        #     if label_no == 5:
        #         break
        #
        # results_payload = OrderedDict()
        # for test in range(len(manufacturer_name)):
        #     testdict = OrderedDict()
        #     testdict["Manufacturer"] = str(manufacturer_name[test])
        #     testdict["Price"] = str(car_price[test])
        #     testdict["Description"] = car_description[test]
        #     results_payload[str(test)] = testdict

        print(results_payload)
        return results_payload