import numpy as np
import pandas as pd
import sys
import os
import csv
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


print("************************************\n")
path = sys.argv[1]
files = [f for f in os.listdir(path) if 'temp1' in f ]

print("************************************\n")


main_df = pd.DataFrame()
all_df = []

mpath = "../metrics/"
epath = "../events/"

event_files = [
    "cache_event_file.log",
    "memory_event_file.log",
    "pipeline_event_file.log",
    "uncore_event_file.log",
    "vm_event_file.log"
]

metric_files = [
    "metric_file.log"
]

all_events = []
for fn in event_files:
    f = open(epath+fn, "r")
    for line in f:
        all_events.append(line.lstrip().rstrip())

for fn in metric_files:
    f = open(mpath+fn, "r")
    for line in f:
        all_events.append(line.lstrip().rstrip())

def func1():
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

def func2():
    for input in files:
        ''' read file and do once more cleaning '''
        infile = "clean_"+input
        outfile = "done_"+input
        f = open(outfile, 'w')

        flag = 1 if 'wi' in input else 0

        print("filename: "+input+", flag"+str(flag)+'\n')

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


def log(s, ok=True):
    if len(sys.argv) >=3:
        ok=False
    if ok == True:
        print(s)

def func3():

    for infile in files:
        infile = 'done_'+infile
        welome ="#################################################################################################\n#\t\tREAD METRICS DONE\t\t\t\t\t\t\t\t\t#\n#################################################################################################"
        log(welome)

        #################################################

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
            if name not in all_events:
                continue
            metrics.append(name)
            timestamps.append(grp['timestamp'].tolist())
            values.append(grp['value'].tolist())
            flags.append(grp['flag'].tolist())


        welome ="#################################################################################################\n#\t\tFILTER BY METRICS DONE\t\t\t\t\t\t\t\t\t#\n#################################################################################################"
        log(welome)

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
        welome ="#################################################################################################\n#\t\tCLEANING DONE\t\t\t\t\t\t\t\t\t#\n#################################################################################################"
        log(welome)


func1()
func2()
func3()

main_df = pd.concat(all_df, axis=0).reset_index(drop=True)
main_df.to_csv('all_df.log')