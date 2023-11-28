import numpy as np
import pandas as pd
import sys
import os
import csv
import locale
import matplotlib.pyplot as plt
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
from threading import Thread
from sklearn.metrics import precision_score, roc_curve, recall_score, auc
from sklearn.metrics import RocCurveDisplay

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

nolabel_df = tmpdf[tmpdf.columns.difference(['flag'])]
labelonly_df = tmpdf['flag']
print(type(nolabel_df))
a = nolabel_df.values
print(a[:,0:1])

tdf = pd.DataFrame(columns=[i for i in range(10)])
k = np.random.uniform(0, 1, size=(1,10))[0]

k[len(k)-1] = 1
print(k)

tdf.loc[0] = k
tdf.loc[len(tdf)] = np.random.uniform(0, 1, size=(1,10))[0]

print(tdf)
# print(tdf)
# print(tmpdf.corr().to_csv("corr.log"))