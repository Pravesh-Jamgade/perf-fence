import numpy as np
import pandas as pd
import sys
import os
import csv

''' choose temp1 (metric) only and merge '''

path = sys.argv[1]

files = [f for f in os.listdir(path) if 'temp1' in f and 'ok_done' in f]

print(files)


main_df = pd.DataFrame()
all_df = []
for f in files:
    df = pd.read_csv(f, sep=',')
    all_df.append(df)
    
main_df = pd.concat(all_df, axis=0).reset_index(drop=True)
print(pd.DataFrame(main_df))
main_df.to_csv('all_df.log')