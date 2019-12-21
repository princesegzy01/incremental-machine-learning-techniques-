from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
from sklearn.decomposition import TruncatedSVD, PCA
import sys
import os
from numpy import array
import time

import nltk                                         #Natural language processing tool-kit

from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer 
import re, string
from nltk.tokenize import word_tokenize 
stemmer = PorterStemmer() 

def preprocessing(document):
    
    # convert to lower case
    document = document.lower()

    # remove numbers
    document = re.sub(r'\d+', '', document) 
    
    # remove punctuation
    translator = str.maketrans('', '', string.punctuation) 
    document =  document.translate(translator) 

    # remove whitespace
    document = " ".join(document.split())

    word_tokens = word_tokenize(document) 

    # remove stop words
    stop_words = set(stopwords.words("english")) 
    word_tokens = word_tokenize(document) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 

    # stemming
    stems = [stemmer.stem(word) for word in filtered_text] 

    # stem document
    # print(filtered_text)

    document = ' '.join(stems)
    return document


categories = os.listdir("datasets/")

news_array = []
news_category = []


for category in categories:
    contents = os.listdir("datasets/"+category)
    for content in contents:

        file_name = "datasets/"+category+"/"+content
        news_content = open(file_name, "r")
        news_content = preprocessing(news_content.read())
        news_array.append(news_content)
        news_category.append(category)



print(" >>>> Done preprocessing >> ")

Y = news_category


vectorizer = TfidfVectorizer(max_features=20000)
X_vec = vectorizer.fit_transform(news_array)


X_vec = X_vec.toarray()

print(X_vec.shape)


X_vec = SelectKBest(mutual_info_classif, k = 15000).fit_transform(X_vec, Y)

# Two features with highest chi-squared statistics are selected 
X_vec = SelectKBest(chi2, k = 10000).fit_transform(X_vec, Y)


pca = PCA(n_components=5000)
X_vec = pca.fit_transform(X_vec) 


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_vec, Y, test_size=0.33, random_state=42)

clf = SGDClassifier(max_iter=10000, tol=1e-3)


start_time = time.time()
clf.fit(X_train, y_train)
elapsed_time = time.time() - start_time
# print(" one shot training : ",elapsed_time)


y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred,y_test)


print(" >>>> One shot : ", acc, " time >> ", elapsed_time)

# clf.partial_fit(X_train, y_train)

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import SGDClassifier

cla = SGDClassifier(max_iter=10000, tol=1e-3)


data_batch = 1000
start = 0

from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np

x_plot = []
y_plot = []


x_plot_mnb = []
y_plot_mnb = []

time_plot = []

# print(X_train)
# print(np.unique(y_train))
for _ in range(0, len(X_train), data_batch):

    x_batch = X_train[start:start+data_batch]
    y_batch = y_train[start:start+data_batch]
    start = start + data_batch
    
    start_time = time.time()
    cla.partial_fit(x_batch, y_batch, classes=['business','education','entertainment','family','politics','sex-relationship','sports'])
    elapsed_time = time.time() - start_time

#     time_plot.append(elapsed_time)


    y_pred = cla.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)  

    x_plot.append(start)
    y_plot.append(accuracy) 

    print(start, " Accuracy >> ", accuracy, " time >> ", elapsed_time) 