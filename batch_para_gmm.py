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
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score
import seaborn as sns


'''
usage:
python3 ../algo.py done.log
'''

def rtn_all_model(test_batch_metrics, all_models, y_test, axs):
    for index in range(len(all_models)):
        name = all_models[index][0]
        gmm = all_models[index][1]
        X_test = test_batch_metrics[index]
        xtest_loglikeli = gmm.score_samples(X_test)#per sample
        thresh = np.percentile(xtest_loglikeli,2)
        testing = pd.DataFrame()
        testing['score'] = xtest_loglikeli
        testing['anomaly'] = testing['score'].apply(lambda x: 1 if x < thresh else 0)
        testing['gnd'] = y_test

        dis= RocCurveDisplay.from_predictions(
            testing['anomaly'],
            testing['gnd'],
            color="darkorange",
            ax=axs[index]
        )
        axs[index].set_title(name+"_model"+str(i))
        axs[index].set_xlabel("FPR")
        axs[index].set_ylabel("TPR")
        axs[index].legend()

def func_main(df, y_train, min_component, loglikeli, sscore, bgmm_loglikeli, bgmm_sscore):
    gmm = GaussianMixture(n_components=min_component, random_state=0).fit(df)#,y_train
    labels = gmm.fit_predict(df)
    loglikeli.append([min_component, gmm.score(df)])
    sscore.append([min_component, silhouette_score(df, labels)])
    
    bgmm = BayesianGaussianMixture(n_components=min_component, random_state=0).fit(df)#,y_train
    labels = bgmm.fit_predict(df)
    bgmm_loglikeli.append([min_component, bgmm.score(df)])
    bgmm_sscore.append([min_component, silhouette_score(df, labels)])


def func_components_for__gmm(df, name, index):
    print("Prcocessing components: " + name)

    loglikeli = []
    sscore = []

    bgmm_loglikeli = []
    bgmm_sscore = []

    all_threads = []
    min_component = 2
    while min_component <= 10:
        t = Thread(target=func_main, args=(df, y_train, min_component, loglikeli, sscore, bgmm_loglikeli, bgmm_sscore))
        t.start()
        all_threads.append(t)
        min_component = min_component + 1
    
    for th in all_threads:
        th.join()
    
    loglikeli.sort()
    sscore.sort()
    bgmm_loglikeli.sort()
    bgmm_sscore.sort()

    logdf = pd.DataFrame(data=loglikeli, columns=['x','y'])
    axs1[index][0].plot(logdf['x'], logdf['y'], color='orange')
    axs1[index][0].set_title(name+"_loglikeli hood")

    ssdf = pd.DataFrame(data=sscore, columns=['x','y'])
    axs1[index][1].plot(ssdf['x'], ssdf['y'], color='orange')
    axs1[index][1].set_title(name+"_score")

    bgmm_logdf = pd.DataFrame(data=bgmm_loglikeli, columns=['x','y'])
    axs1[index][0].plot(bgmm_logdf['x'], bgmm_logdf['y'], color='limegreen')
    # axs[index][0].set_title(name+"_bgmm_loglikeli hood")

    bgmm_ssdf = pd.DataFrame(data=bgmm_sscore, columns=['x','y'])
    axs1[index][1].plot(bgmm_ssdf['x'], bgmm_ssdf['y'], color='limegreen')
    # axs[index][1].set_title(name+"_bgmm_score")


'''
about: Training GMM
param: name, X_train, y_train=None (default), all_models
'''
def func_process_using_gmm(name, X_train, y_train, all_models):
    k = 4
    gmm = GaussianMixture(n_components=k, random_state=0).fit(X_train) 
    all_models.append((name,gmm))
    # plt.show()


def log(s, ok=True):
    if len(sys.argv) >=3:
        ok=False
    if ok == True:
        print(s)





if __name__ == "__main__":

    fileName = sys.argv[1]
    newdf = pd.read_csv(fileName)
    print(newdf.head(3))

    x = newdf.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    tmpdf = pd.DataFrame(x_scaled, columns=newdf.columns)
    print(tmpdf.head(3))

    welome ="#################################################################################################\n#\t\tNORMALIZE DONE\t\t\t\t\t\t\t\t\t#\n#################################################################################################"
    log(welome)

    nolabel_df = tmpdf[tmpdf.columns.difference(['flag'])]
    labelonly_df = tmpdf['flag']

    X = np.array(nolabel_df.values)
    y = np.array(labelonly_df.values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44)

    STEP = 100
    train_batch_metrics = []
    test_batch_metrics = []
    no_batches = X_train.shape[1]
    no_batches = no_batches // 100

    step = 0
    for i in range(no_batches):
        end = step + STEP
        print(str(step) +" -- "+ str(end))
        if i == no_batches-1:
            train_batch_metrics.append(X_train[:, step:end+1])
            test_batch_metrics.append(X_test[:, step:end+1])
        else:
            train_batch_metrics.append(X_train[:, step:end])
            test_batch_metrics.append(X_test[:, step:end])
        step = end

    fig1, axs1 = plt.subplots(5, 2, layout="constrained")
    fig2, axs2 = plt.subplots(5, 1, layout="constrained", sharex=True, sharey=True)
    fig3, axs3 = plt.subplots(5, 1, layout="constrained", sharex=True, sharey=True)

    step = 0
    end = 0
    all_models = []

    check_component = str(input("Check components?(y/n)"))
    for index in range(len(train_batch_metrics)):
        end = step + STEP
        name = str(step) + "_" + str(end)
        step = end

        df = train_batch_metrics[index]
        test_df = test_batch_metrics[index]

        if check_component == 'y':
            func_components_for__gmm(df, name, index)
        
        print("Prcocessing GMM: " + name)
        func_process_using_gmm(name, df, y_train, all_models)

    rtn_all_model(test_batch_metrics, all_models, y_test, axs2)


    ## TESTING

    # test_df = pd.read_csv('ok_done_wo.a2.temp1.log')
    # val = np.random.uniform(0, 1, size=(1,test_df.shape[1]))[0]
    # val[len(val)-1] = 0
    # test_df.loc[len(test_df)] = val
    
    # x = test_df.values #returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # tmpdf = pd.DataFrame(x_scaled, columns=test_df.columns)
    
    # nolabel_df = tmpdf[tmpdf.columns.difference(['flag'])]
    # labelonly_df = tmpdf['flag']

    # X_test = nolabel_df.values
    # y_test = labelonly_df.values

    # test_batch_metrics = []
    # step = 0
    # for i in range(no_batches):
    #     end = step + STEP
    #     print(str(step) +" -- "+ str(end))
    #     test_batch_metrics.append(X_test[:, step:end])
    #     step = end

    # rtn_all_model(test_batch_metrics, all_models, y_test, axs3)
    plt.show()