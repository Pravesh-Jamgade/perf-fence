import numpy as np
import pandas as pd
import sys
import os
import csv

filePath = sys.argv[1]

f=open("temp.log", "r")
out=open("output.log", "w")

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

with open('output.log', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i = 10
    for row in spamreader:
        if i>0:
            print(row)
        i=i-1