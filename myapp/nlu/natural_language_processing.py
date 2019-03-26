# Natural Language Processing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
dataset = pd.read_csv(str(dir_path)+'/master_data.tsv', delimiter = '\t', quoting = 3, error_bad_lines=False)

dataset_total_rows = len(dataset)

'''
Elasticsearch approach to NLU : https://www.elastic.co/blog/text-classification-made-easy-with-elasticsearch
'''

'''
Spacy and textacy is preferrable for modern nlu.. Spacy has inbuild nlu pipeline which looks like this... -> https://cdn-images-1.medium.com/max/1200/1*zHLs87sp8R61ehUoXepWHA.png
https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e
'''

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.tokenize import RegexpTokenizer

TOKENIZER = RegexpTokenizer('(?u)\W+|\$[\d\.]+|\S+')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import  datetime
def Logger(msg):
    # Log an error message
    with open("E:\\python\\jangoD invisible\\myproject5\\myproject3\\myapp\\nlu\\logger.txt", "a") as myfile:
        myfile.write(str(datetime.datetime.now()) +" "+ str(msg) + "\n")
def takeInput(data):
    corpus = []
    # Predicting the Test set results
    prediction_list = []


    for i in range(0, dataset_total_rows):
        print(dataset['sentance'][i])
        review = re.sub('[^a-zA-Z]', ' ', dataset['sentance'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
        print(review)


    new_text_by_user = data
    review2 = re.sub('[^a-zA-Z]', ' ', new_text_by_user)
    review2 = review2.lower()
    review2 = review2.split()
    ps2 = PorterStemmer()
    review2 = [ps2.stem(word) for word in review2 if not word in set(stopwords.words('english'))]
    review2 = ' '.join(review2)
    corpus.append(review2)


    Logger("Corpus length "+ str(len(corpus)))



    # Creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 1500)
    X = cv.fit_transform(corpus).toarray()


    X_formatted = np.array([X[-1,:]])



    y = dataset.iloc[:, 1].values




    #categorical data
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)


    X_except_last_row = X[:-1, :]
    # Splitting the dataset into the Training set and Test set
    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X_except_last_row, y, test_size = 0.20, random_state = 0)
    # Fitting Naive Bayes to the Training set


    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_except_last_row, y)

    import pickle
    # save the classifier
    with open('my_dumped_classifier.pkl', 'wb') as fid:
        pickle.dump(classifier, fid)


    # load it again
    with open('my_dumped_classifier.pkl', 'rb') as fid:
        classifier_loaded = pickle.load(fid)


    y_pred1 = classifier.predict(X_formatted)
    y_pred2 = classifier.predict_log_proba(X_formatted)
    y_pred3 = classifier.predict_proba(X_formatted)
    prediction_list.append(y_pred1)
    prediction_list.append(y_pred2)
    prediction_list.append(y_pred3)


    # Making the Confusion Matrix
    #from sklearn.metrics import confusion_matrix
    #cm = confusion_matrix(y_test, y_pred1)

    name_entity = ""
    Logger("ypred 1 0 === " + str(y_pred1[0]))
    if y_pred1[0] == 2:
        def preprocess(sent):
            sent = nltk.word_tokenize(sent)
            sent = nltk.pos_tag(sent)
            return sent
        sent2 = preprocess(data)
        for x in range(len(sent2)):
            if sent2[x][1] == "NNP":
                if sent2[x][0] != " ":
                    name_entity = sent2[x][0]

        if name_entity == " " :
            prediction_list.append("<br>NER couldn't recognize the entity. :( <br>Traceback Array<br>"+str(sent2)+"<br>"+str(name_entity))
        else:
            prediction_list.append(str(name_entity))

        Logger("prediction list if " + str(sent2))

        Logger("nltk entity ----> " + str(sent2))
    else:
        prediction_list.append("<!_name>")
        Logger("prediction list else " + str(prediction_list))
    Logger("prediction list count " + str(len(prediction_list)))

    return prediction_list

