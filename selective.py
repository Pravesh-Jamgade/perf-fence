import os
import sys
import shlex, subprocess
import argparse
from multiprocessing import Process
from subprocess import check_output, CalledProcessError
import yaml

path = "select_input.log"

with open(path, "r") as file:
    input = yaml.safe_load(file)

evt_cmd = ""

if len(sys.argv) < 3:
    print("missing arguments\n")
    exit(0)

for evt in input['event']:
    evt_cmd = evt_cmd+evt.lstrip().rstrip()+","
evt_cmd = evt_cmd[0:len(evt_cmd)-1]

print("sampling_interval: "+str(sys.argv[1]))
print("no. of intervals: "+str(sys.argv[2]))

fire_event = "perf stat --post /mnt/B/sem3/mss/project/post.sh -e {} -o /mnt/B/sem3/mss/project/select_output.log -I {} --interval-count {}".format( evt_cmd, sys.argv[1], sys.argv[2])

print(fire_event)
try:
    check_output(shlex.split(fire_event))
except CalledProcessError as e:
    print("exception: " + str(os.getpid()))
print("end: " + str(os.getpid()))