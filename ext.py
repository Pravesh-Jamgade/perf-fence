import os
import sys

path = "temp.log"
f = open(path, 'r')

keep_looking = False
for line in f:
    
    line = line.lstrip()
    line = line.rstrip()
   
    if '[' in line:
        keep_looking = True
    if ']' in line:
        keep_looking = False
        continue

    if '[' in line and ']' in line:
        keep_looking = False
        continue
    
    if keep_looking == True:
        continue
    print(line)