from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_extraction.text import CountVectorizer

categories = ['talk.religion.misc','comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='all',categories=categories)


X, Y = newsgroups_train.data, newsgroups_train.target

# print(len(X))

cv = CountVectorizer(max_df=0.95, min_df=2,max_features=10000,stop_words='english')
X_vec = cv.fit_transform(X)




# print(X_vec.shape)

# print(len(X_vec.toarray()))

# reduc = mutual_info_classif(X_vec, Y, discrete_features=True)
# X_new = SelectPercentile(mutual_info_classif, percentile=10).fit_transform(X_vec, Y)

X_new = X_vec


# # print(X_new.toarray()[0])
# res = dict(zip(cv.get_feature_names(),mutual_info_classif(X_vec, Y, discrete_features=True)))
# print(res)
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.33, random_state=42)

clf = SGDClassifier(max_iter=1000, tol=1e-3)


# import time

# start_time = time.time()


# clf.fit(X_train, y_train)

# elapsed_time = time.time() - start_time
# print("time 1 : ", elapsed_time)


# y_pred = clf.predict(X_test)


# print(" <<>> ", X_train.shape)


# from sklearn.metrics import accuracy_score
# pred = accuracy_score(y_pred,y_test)

# print(" >>>> ", pred)

# start_time2 = time.time()
# clf.partial_fit(X_train, y_train)
# elapsed_time = time.time() - start_time2
# print(" time 2 : ",elapsed_time)


from sklearn.linear_model import SGDClassifier
cla = SGDClassifier()
performance = []

data_batch = 50
start = 0

print(y_train[1])


# for i in range(0, X_train.shape[0] , data_batch):
# for i in range(0, X_train.shape[0] , data_batch):

    # x_batch = X_train[i:data_batch]
    # y_batch = y_train[0:data_batch]

    # print(i, " >> " , y_train[0:50])
    
    # print(len(y_train))
    # print(x_batch, " -- ", y_batch)
#     cla.partial_fit(x_batch, y_batch, classes=[0, 1, 2])
#     conf = confusion_matrix(y_test, cla.predict(x_test))
#     performance.append(np.diag(conf) / np.sum(conf, axis=1))
    