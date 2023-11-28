
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
import os
import sys
import shlex, subprocess
import argparse
from multiprocessing import Process
from subprocess import check_output, CalledProcessError


fig1, axs1 = plt.subplots(5, 2, layout="constrained")
fig2, axs2 = plt.subplots(5, 1, layout="constrained", sharex=True, sharey=True)
fig3, axs3 = plt.subplots(5, 1, layout="constrained", sharex=True, sharey=True)

STEP = 100
THRESH = 0

def func_helper(x, th_map):
    x['anomaly'] = x['score'].apply(lambda t:t <= th_map[x.name])
    return x

def func_test_samples(main_df, fit_models, single_gmm):
    X = np.array(main_df.values)

    no_batches = main_df.shape[1]
    no_batches = no_batches // 100

    test_batch_metrics = []
    step = 0
    for i in range(no_batches):
        end = step + STEP
        print(str(step) +" -- "+ str(end))
        # if i == no_batches-1:
        #     test_batch_metrics.append(main_df[:, step:end+1])
        # else:
        test_batch_metrics.append(X[:, step:end])
        step = end

    for index in range(len(fit_models)):
        model = fit_models[index][1]
        test_batch = test_batch_metrics[index]
        xtest_loglikeli = model.score_samples(test_batch)

        THRESH = np.percentile(xtest_loglikeli, 3)

        testing = pd.DataFrame()
        testing['score'] = xtest_loglikeli
        testing['anomaly'] = testing['score'].apply(lambda x: 1 if x < THRESH else 0)
        testing['prediction'] = model.predict(test_batch)

        # th_map = {}
        # for name, grp in testing.groupby(['prediction']):
        #     THRESH = np.percentile(grp['score'], 3)
        #     th_map[name] = THRESH        

        # testing = testing.groupby('prediction').apply(lambda x: func_helper(x, th_map))
        
        counts = testing['anomaly'].value_counts().to_dict()
        print("[Batch GMM] sizeof test: " + str(testing.shape))
        print("[Batch GMM] safe(%)" + str(counts[0]*100/ (counts[1]+counts[0])))
    
    data_likeli = single_gmm.score_samples(main_df)
    THRESH = np.percentile(data_likeli, 3)
    testing = pd.DataFrame()
    testing['score'] = data_likeli
    testing['anomaly'] = testing['score'].apply(lambda x: 1 if x < THRESH else 0)
    testing['prediction'] = single_gmm.predict(main_df)
    counts = testing['anomaly'].value_counts().to_dict()
    print("[Single GMM] sizeof test: " + str(testing.shape))
    print("[Single GMM] safe(%)" + str(counts[0]*100/ (counts[1]+counts[0])))


def func3(files):
    all_df = []
    for infile in files:
        infile = 'done_'+infile
        
        cols = ['timestamp', 'value', 'metric', 'flag']
        df = pd.read_csv(infile, sep='\\')
        df.columns = cols

        ''' attributes on x and samples on y '''

        metrics = []
        timestamps = []
        values = []
        flags = []

        for name, grp in df.groupby('metric'):
            name = name.lstrip().rstrip()
            if name not in check_for_metrics:
                continue
            metrics.append(name)
            timestamps.append(grp['timestamp'].tolist())
            values.append(grp['value'].tolist())
            flags.append(grp['flag'].tolist())

        print("metrics: "+str(len(metrics)))
        print("timestamps: "+str(len(timestamps)))
        print("values: "+str(len(values)))

        # make sure all metrics values are available at each timestep, if len of any timestamp of metric is not equal to others, that mean
        # either curr metric is calculated more/less number of timestamp than others 
        uni = []
        for val in timestamps:
            if len(val) not in uni:
                uni.append(len(val))

        # finding min timestamp avail for all metrics
        min = uni[0]
        for k in uni:
            if min>k:
                min=k
        # #####debug
        # print("min len: ", min)

        # assigning each metric its value
        dic = {str(metrics[i]): values[i] for i in range(len(metrics)) }
        dic['flag'] = flags[0][0:min]#as flag is constant across for wo and wi files

        for k in dic:
            dic[k] = dic[k][0:min]
        # #####debug
        # for k in dic:
        #     print(k + "," + str(len(dic[k])))


        newdf = pd.DataFrame.from_dict(dic)
        # #####debug
        # print(newdf.head(3))
        print(newdf.shape)

        newdf.to_csv("ok_"+infile)

        all_df.append(newdf)
       
    main_df = pd.DataFrame()
    main_df = pd.concat(all_df, axis=0).reset_index(drop=True)
    main_df = main_df.drop(columns=['flag'])

    x = main_df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    main_df = pd.DataFrame(x_scaled, columns=main_df.columns)

    # for req in check_for_metrics:
    #     if req not in main_df.columns:
    #         print("[+]"+req)
    
    # for avail in main_df.columns:
    #     if avail not in check_for_metrics:
    #         print("[-]"+avail)
    # print('\n')
    # print(check_for_metrics)
    # print('\n')
    # print(list(main_df.columns))

    # print('\n')
    # print(len(check_for_metrics))
    # print('\n')
    # print(len(list(main_df.columns)))
    
    return main_df


def func2(files):
    for input in files:
        ''' read file and do once more cleaning '''
        infile = "clean_"+input
        outfile = "done_"+input
        f = open(outfile, 'w')

        flag = 1 if 'wi' in input else 0

        print("filename: "+input+", flag"+str(flag)+'\n')

        f.write("A\\B\\C\\D\n")
        # write to outfile with appended attack flag, read from infile
        with open(infile, newline='') as csvfile:
            for row in csvfile:
                ele = row.split('\\')
                
                # test if it gives exception, if it is a non-digit variable then except
                # if comma separated numbers, then they will be changed to int
                try:
                    temp = locale.atoi(ele[1])
                except:
                    continue
                
                # print(ele[0].lstrip().rstrip() + ',' + str(temp)+ ',' + ele[2].lstrip().rstrip() +'\n')
                f.write(ele[0].lstrip().rstrip() + '\\' + str(temp)+ '\\' + ele[2].lstrip().rstrip() + '\\'+ str(flag) +'\n')


def func1(files):
    for infile in files:
        outfile = "clean_"+infile
        f=open(infile, "r")
        out=open(outfile, "w")
        for line in f:
                word = ""
                col = 2
                for w in line.split(" "):
                    if col<0:
                        break
                    w = w.lstrip().rstrip()
                    if w == " " or w == "":
                        continue
                    if w == '#' or w == '(':
                        break
                    word = word+w+" \\" 
                    col = col - 1       
                word = word + '\n'
                out.write(word)


def func_preprocess_samples(file_name):
    files = [f for f in os.listdir('.') if file_name in f ]
    func1(files)
    func2(files)
    return func3(files)


def rtn_train_model(df, y_train, min_component, loglikeli, sscore, bgmm_loglikeli, bgmm_sscore):
    gmm = GaussianMixture(n_components=min_component, random_state=0).fit(df,y_train)
    labels = gmm.fit_predict(df)
    loglikeli.append([min_component, gmm.score(df)])
    sscore.append([min_component, silhouette_score(df, labels)])
    
    bgmm = BayesianGaussianMixture(n_components=min_component, random_state=0).fit(df)#,y_train
    labels = bgmm.fit_predict(df)
    bgmm_loglikeli.append([min_component, bgmm.score(df)])
    bgmm_sscore.append([min_component, silhouette_score(df, labels)])


def rtn_all_model(test_batch_metrics, all_models, y_test, axs):
    for index in range(len(all_models)):
        name = all_models[index][0]
        gmm = all_models[index][1]
        X_test = test_batch_metrics[index]

        xtest_loglikeli = gmm.score_samples(X_test)#per sample
        # THRESH = np.percentile(xtest_loglikeli,2)

        testing = pd.DataFrame()
        testing['score'] = xtest_loglikeli
        testing['anomaly'] = testing['score'].apply(lambda x: 1 if x < THRESH else 0)
        testing['gnd'] = y_test
        testing['prediction'] = gmm.predict(X_test)

        # th_map = {}
        # for nm, grp in testing.groupby(['prediction']):
        #     THRESH = np.percentile(grp['score'], 3)
        #     th_map[nm] = THRESH        

        # testing = testing.groupby('prediction').apply(lambda x: func_helper(x, th_map))

        dis= RocCurveDisplay.from_predictions(
            testing['anomaly'],
            testing['gnd'],
            color="darkorange",
            ax=axs[index]
        )
        axs[index].set_title(name+"_model"+str(index))
        axs[index].legend()


def rtn_components_for__gmm(df, name, index, y_train):
    print("Prcocessing components: " + name)

    loglikeli = []
    sscore = []

    bgmm_loglikeli = []
    bgmm_sscore = []

    all_threads = []
    min_component = 2
    while min_component <= 10:
        t = Thread(target=rtn_train_model, args=(df, y_train, min_component, loglikeli, sscore, bgmm_loglikeli, bgmm_sscore))
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


def rtn_process_using_gmm(name, X_train, y_train, all_models):
    k = 4
    gmm = GaussianMixture(n_components=k, random_state=0).fit(X_train, y_train) 
    all_models.append((name,gmm))


def func_init_training():
    fileName = args.training_data
    newdf = pd.read_csv(fileName)

    x = newdf.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    tmpdf = pd.DataFrame(x_scaled, columns=newdf.columns)

    check_for_metrics = []
    check_for_metrics = check_for_metrics + tmpdf.columns.difference(['flag', 'Unnamed: 0']).tolist()

    nolabel_df = tmpdf[tmpdf.columns.difference(['flag'])]
    labelonly_df = tmpdf['flag']

    X = np.array(nolabel_df.values)
    y = np.array(labelonly_df.values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44)

    train_batch_metrics = []
    test_batch_metrics = []
    no_batches = X_train.shape[1]
    no_batches = no_batches // 100

    step = 0
    for i in range(no_batches):
        end = step + STEP
        print(str(step) +" -- "+ str(end))
        # if i == no_batches-1:
        #     train_batch_metrics.append(X_train[:, step:end+1])
        #     test_batch_metrics.append(X_test[:, step:end+1])
        # else:
        train_batch_metrics.append(X_train[:, step:end])
        test_batch_metrics.append(X_test[:, step:end])
        step = end
    
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
            rtn_components_for__gmm(df, name, index, y_train)
        
        print("Prcocessing GMM: " + name)
        rtn_process_using_gmm(name, df, y_train, all_models)

    rtn_all_model(test_batch_metrics, all_models, y_test, axs2)
    
    single_gmm = GaussianMixture(n_components=4, random_state=0).fit(X_train, y_train)
    return all_models, check_for_metrics, single_gmm


def func_collect_samples(fire_args):
    print("start: " + str(os.getpid()))
    try:
        check_output(fire_args)
    except CalledProcessError as e:
        print("exception: " + str(os.getpid()))
    print("end: " + str(os.getpid()))

def func_log_process_stage(msg):
    print("#############################################")
    print(msg)
    print("#############################################")


parser = argparse.ArgumentParser(
                    prog='PEMtool',
                    description='PMU Events Metric collector tool',
                    epilog='EPILOG')


parser.add_argument('-s', "--sampling_interval", help="sample period (ms)")
parser.add_argument('-c', "--interval_count", help="no. of interval's to collect data for")
parser.add_argument('-t', "--training_data", help="training data")
args = parser.parse_args()

root_path = "."
evt_path = "../events"

event_files = [
    "cache_event_file.log",
    "memory_event_file.log",
    "pipeline_event_file.log",
    "uncore_event_file.log",
    "vm_event_file.log"
]

all_target_file_obj = []
for file_name in event_files:
    f = open(evt_path+'/'+file_name, 'r')
    all_target_file_obj.append(f)

sampling_interval = int(args.sampling_interval)
interval_count = int(args.interval_count)

print("sampling_interval: ", sampling_interval)
print("interval_count: ", interval_count)

evt_cmd = ""
all_events = []
for f in all_target_file_obj:
    for evt in f:
        evt_cmd = evt_cmd + evt.lstrip().rstrip() + ","
        all_events.append(evt.lstrip().rstrip() )
evt_cmd = evt_cmd[0:len(evt_cmd)-1]

fit_models, check_for_metrics, single_gmm = func_init_training()

for i in range(1):
    file_name = "ver.{}.online.temp1.log".format(i)
    fire_event = "perf stat --post /mnt/B/sem3/mss/project/post.sh -e {} -o /mnt/B/sem3/mss/project/temp_online/{} -I {} --interval-count {}".format( evt_cmd,file_name, sampling_interval, interval_count)
    args1 = shlex.split(fire_event)
    
    func_collect_samples(args1)
    func_log_process_stage("[COLLECTION] {} ...ok".format(file_name))

    main_df = func_preprocess_samples(file_name)
    func_log_process_stage("[PREPROCESSING] {} ...ok".format(file_name))

    func_test_samples(main_df, fit_models, single_gmm)
    func_log_process_stage("[PREDICTION] {} ...ok".format(file_name))

    
    # plt.show()



