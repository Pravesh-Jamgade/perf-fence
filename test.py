import os
import sys
import shlex, subprocess
import argparse
from multiprocessing import Process

def func1():
    for i in range(10):
        r = str(i) + "_name"
        try:
            open(r, 'r')
            subprocess.Popen(shlex.split("rm -rf temp_dir/{}".format(r)))
        except:
            break

def func2():
    i = 15594869
    while True:
        i = i +1
        r = str(i) + "_name"
        try:
            open(r, 'r')
            subprocess.Popen(shlex.split("rm -rf temp_dir/{}".format(r)))
            print("done..{}".format(r))
        except:
            print("not")

proc1 = Process(target=func2)
proc1.start()
proc1.join()

