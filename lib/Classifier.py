from nltk import WordNetLemmatizer, LancasterStemmer
from nltk.corpus import stopwords
import pandas as pd
import operator
import logging

logger = logging.getLogger()


class Naive_bayes_classifier():
    def __init__(self):
        # self.UsedCarsDS = pd.read_csv(
        #     "E:/Data Mining/Dataset/smaller dataset/craigslistVehicles/3.7 dataset/craigslistVehiclesCheck.csv")
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
            logger.info("Classifier: Initializing inverted Index")
            self.total_rows = len(self.UsedCarsDS)
            # self.total_rows = 200
            logger.info("Classifier: Total rows : {}".format(self.total_rows))
            for idx in self.UsedCarsDS.index:
                # if idx == self.total_rows:
                #     break
                words = str(self.UsedCarsDS.get_value(idx, 'desc'))
                paint_color = str(self.UsedCarsDS.get_value(idx, 'paint_color'))
                if paint_color:
                    words += " "+paint_color

                type_of_car = str(self.UsedCarsDS.get_value(idx, 'type'))
                if type_of_car:
                    words += " "+type_of_car

                manufacturer = str(self.UsedCarsDS.get_value(idx, 'manufacturer'))
                if manufacturer:
                    words += " "+manufacturer

                price_car = str(self.UsedCarsDS.get_value(idx, 'price'))
                if price_car:
                    words += " " + price_car

                condition_car = str(self.UsedCarsDS.get_value(idx, 'condition'))
                if condition_car:
                    words += " " + condition_car

                unique_words = (self.tokenize(words))
                unique_words_set = set(unique_words)
                self.unique_word_set_global = self.unique_word_set_global.union(unique_words_set)
                label = self.UsedCarsDS.get_value(idx, 'type')

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
                            temp_dict[word] = 1
                        else:
                            temp_dict[word] = temp_dict[word] + 1
                        self.class_wise_dict[label] = temp_dict
                    else:
                        temp_dict = {word: 1}
                        self.class_wise_dict[label] = temp_dict
                        self.labels.append(label)
            logger.info("Classifier: total number of rows iterated : {}".format(idx))
        except Exception as e:
            logger.error("Classifier: Exception occurred while initializing Classifier : {}".format(e))
            logger.info("Classifier Exception: total number of rows iterated : {}".format(idx))

    # we tokenize to remove all the stopwords and return a set of words
    def tokenize(self, description):

        filtered = []
        # dont process NaN or Null values
        if pd.isnull(description):
            return filtered, filtered
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

            return filtered_final

    def classify_dataset(self, input_string, SearchObj):
        try:
            input_string_tokenized = self.tokenize(input_string)
            validate = False

            # if not input_string_tokenized:
            #     validate = False
            # else:
            #     for each_word in input_string_tokenized:
            #         if each_word in self.unique_word_set_global:
            #             validate = True
            #             break
            #
            # if not validate:
            #     logger.info("Either input string is empty or none of these words exist in Data-set description")
            #     return []

            final_dict = {}
            sum_all_label_probability = 0
            logger.debug("Classifier: Labels : {}".format(self.labels))
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
                final_label_percentage_ranked.append(round((tup[1] / sum_all_label_probability) * 100, 2))

            search_results = SearchObj.search_dataset_with_label(input_string, final_label_ranked[0])

            results_payload = [final_label_ranked, final_label_percentage_ranked, search_results]

            logger.debug("Classifier: Classifier Final result payload")
            logger.debug(results_payload)
            return results_payload
        except Exception as e:
            logger.error("Classifier: Exception occurred in Classifier : {}".format(e))
