import numpy as np
import pandas as pd
import sys
import os
import csv
import locale

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

'''
usage:
python3 ../algo.py done.log
'''

def log(s, ok=True):
    if len(sys.argv) >=3:
        ok=False
    if ok == True:
        print(s)

fileName = sys.argv[1]
newdf = pd.read_csv(fileName)
print(newdf.head(3))

import pandas as pd
from sklearn import preprocessing

x = newdf.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
tmpdf = pd.DataFrame(x_scaled, columns=newdf.columns)
print(tmpdf.head(3))

welome ="#################################################################################################\n#\t\tNORMALIZE DONE\t\t\t\t\t\t\t\t\t#\n#################################################################################################"
log(welome)


from sklearn.model_selection import train_test_split


nolabel_df = tmpdf[tmpdf.columns.difference(['flag'])]
labelonly_df = tmpdf['flag']

X = np.array(nolabel_df.values)
y = np.array(labelonly_df.values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44)

''' APPLY GMM '''
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

predicted_labels = []
loglikeli_data = []
model_score = []

# dim = tmpdf.shape
# print(dim)

import seaborn as sns
import matplotlib.pyplot as plt

############## UNCOMMENT to see model performance and avg log likelihood over k number of components
# min_comp = 100
# k = 3
# while k <= min_comp:
#     gm = GaussianMixture(n_components=k, random_state=0).fit(X_train)
#     labels = gm.predict(X_train)
#     predicted_labels.append(labels)
#     loglikeli_data.append([k,gm.score(X_train)])
#     sscore = silhouette_score(X_train, labels)
#     model_score.append([k,sscore])
#     k = k +1

# log_df = pd.DataFrame(loglikeli_data, columns=['x','y'])
# print(log_df.head(2))
# sns.lineplot(log_df['x'], log_df['y'])
# plt.show()

# sscode_df = pd.DataFrame(model_score, columns=['x','y'])
# sns.lineplot(sscode_df['x'], sscode_df['y'])
# plt.show()

compo = 5#int(input("components="))

gm = GaussianMixture(n_components=compo, random_state=0).fit(X_train, y_train)

xtrain_loglikeli = gm.score_samples(X_train)#per sample
thresh = np.percentile(xtrain_loglikeli,2)
training = pd.DataFrame()
training['score'] = xtrain_loglikeli
training['anomaly'] = training['score'].apply(lambda x: 1 if x < thresh else 0)
training['gnd'] = y_train

xtest_loglikeli = gm.score_samples(X_test)#per sample
thresh = np.percentile(xtest_loglikeli,2)
testing = pd.DataFrame()
testing['score'] = xtest_loglikeli
testing['anomaly'] = testing['score'].apply(lambda x: 1 if x < thresh else 0)
testing['gnd'] = y_test

# sns.histplot(testing['score'], bins=10, alpha=0.8)
# plt.axvline(x=thresh, color='orange')
# plt.show()

# sns.lineplot(training.index, training['anomaly'], color='red')
# sns.lineplot(training.index, training['gnd'], color='green')
# plt.show()

# sns.lineplot(testing.index, testing['anomaly'], color='red')
# sns.lineplot(testing.index, testing['gnd'], color='green')
# plt.show()

from sklearn.metrics import precision_score, roc_curve, recall_score, auc

print("original shape: " + str(tmpdf.shape))
print("train shape: " + str(X_train.shape))
print("testing shape: " + str(X_test.shape))

a=precision_score(training['gnd'], training['anomaly'], average='macro')
print("training preicision: "+str(a))
b=precision_score(testing['gnd'], testing['anomaly'], average='macro')
print("testing preicision: "+str(b))

# a=recall_score(training['gnd'], training['anomaly'], average='macro')
# print("training recall: "+str(a))
# b=recall_score(testing['gnd'], testing['anomaly'], average='macro')
# print("testing recall: "+str(b))

fpr, tpr, thresholds = roc_curve(testing['gnd'], testing['anomaly'])
print(fpr, tpr, thresholds, auc(fpr, tpr))
fpr, tpr, thresholds = roc_curve(training['gnd'], training['anomaly'])
print(fpr, tpr, thresholds, auc(fpr, tpr))

from sklearn import metrics
from sklearn.cluster import DBSCAN

# clustering = DBSCAN(eps=3, min_samples=5).fit(X_train)
# pred = []
# for i in clustering.labels_:
#     if i == -1:
#         pred.append(1)
#     else:
#         pred.append(0)
# print("dbscan training precision: " + str(precision_score(y_train, pred, average='macro')))
# print("dbscan training recall: " + str(recall_score(y_train, pred, average='macro')))

# clustering = DBSCAN(eps=3, min_samples=5).fit(X_test)
# pred = []
# for i in clustering.labels_:
#     if i == -1:
#         pred.append(1)
#     else:
#         pred.append(0)
# print("dbscan testing precision: " + str(precision_score(y_test, pred, average='macro')))
# print("dbscan testing recall: " + str(recall_score(y_test, pred, average='macro')))
# #####################
# # min_dist = 100
# # k = 2
# # while k <= min_dist:



# gm = GaussianMixture(n_components=compo, random_state=0)
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# n_classes = len(np.unique(y))
# X = np.concatenate([X, random_state.randn(n_samples,n_features)], axis=1)
# (
#     X_train,
#     X_test,
#     y_train,
#     y_test,
# ) = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
# y_score = gm.fit(X_train, y_train).predict_proba(X_test)

# print(y_score)
# from sklearn.preprocessing import LabelBinarizer

# label_binarizer = LabelBinarizer().fit(y_train)
# y_onehot_test = label_binarizer.transform(y_test)
# print(y_onehot_test.shape)  # (n_samples, n_classes)   

# print(label_binarizer.transform([1]))
# class_of_interest = 1
# class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
# print(class_id)

import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(
    testing['anomaly'],
    testing['gnd'],
    color="darkorange",
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("")
plt.legend()
plt.show()