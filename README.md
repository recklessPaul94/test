# test

This is my readme file to walkthrough all the three phases of my project.

# Flask Application

My FlaskApplication.py is the soul of the project. This is the file that is run on server start-up to initalize all the classes and render the homepage


# Phase 1 Search

From the FlaskApplication.py we first initalize the inverted index of all the rows in the dataset 

We also keep a record of all the documents it been in

When a user gives an input query we first process the input from the resultspage controller in FlaskApplication.py and call the search dataset methond in class TfIdfSearch.py

We calculate the TF-IDF score for the input query and then compare it with the TF-IDF scores of all the rows of the dataset.

After that we use cosine similarity to get the similarity between two vectors and rank them accordingly

We then get the relevant data from these top ranked rows and return it back to the UI


# Phase 2 Classifier

From the FlaskApplication.py we first initalize the Class wise inverted index of all the rows in the dataset 

When a user gives an input query we first process the input from the classifierresultspage controller in FlaskApplication.py and call the classify method in class Classifier.py

In Classifier.py we calculate the Naive-Bayes score of each of the labels by using the Class wise inverted index that we created earlier

We rank them accordingly to the most probable labels

We then search for cars in the dataset by performing TF-IDF search again but this time only retreiving records of the highest recorded labels

We then return this to the front end


# Phase 3 Image Captioning


# Image search with text

From the FlaskApplication.py we first initalize the inverted index of all the rows in the dataset wit the captions that we created using the Attention Based Model

We also keep a record of all the documents it been in

When a user gives an input query we first process the input from the imagecaptionresultspage controller in FlaskApplication.py and call the search dataset methond in class Image_Captioning.py

We calculate the TF-IDF score for the input query and then compare it with the TF-IDF scores of all the rows of the dataset.

After that we use cosine similarity to get the similarity between two vectors and rank them accordingly

We then get the relevant data from these top ranked rows and return it back to the UI


# Image search with Image input

From the FlaskApplication.py we first load all the data that we collected from training the model on google colab.

We then take an input image from the user and save it in a local directory and retreive the image path to process further

We then take this image and pass it to the evaluate method in ImageEvaluator.py and process it to create its captions

We pass this caption in the Image Search with text method and fetch all similar images based on TF-IDF score of the captions

After that we use cosine similarity to get the similarity between two vectors and rank them accordingly

We then get the relevant data from these top ranked rows and return it back to the UI