import os
import sys
import shlex, subprocess
import argparse
from multiprocessing import Process
from subprocess import check_output, CalledProcessError

parser = argparse.ArgumentParser(
                    prog='PEMtool',
                    description='PMU Events Metric collector tool',
                    epilog='EPILOG')


parser.add_argument('-s', "--sampling_interval", help="sample period (ms)")
parser.add_argument('-p', "--process_id", help="process id, if not given then system wide data collection")
parser.add_argument('-c', "--interval_count", help="no. of interval's to collect data for")

args = parser.parse_args()


def fire_in_the_hole(fire_args):
    print("start: " + str(os.getpid()))
    try:
        check_output(fire_args)
    except CalledProcessError as e:
        print("exception: " + str(os.getpid()))
    print("end: " + str(os.getpid()))


root_path = "."
evt_path = "./events"
met_path = "./metrics"

log_file = open(root_path+'/logfile.log', 'w')
evt_file = open(evt_path+'/event_file.log', "r")
cache_evt_file = open(evt_path+'/cache_event_file.log', "r")
float_evt_file = open(evt_path+'/float_event_file.log', "r")
frontend_evt_file = open(evt_path+'/frontend_event_file.log', "r")
memory_evt_file = open(evt_path+'/memory_event_file.log', "r")
hwinter_evt_file = open(evt_path+'/hwinter_event_file.log', "r")
pipeline_evt_file = open(evt_path+'/pipeline_event_file.log', "r")
uncore_evt_file = open(evt_path+'/uncore_event_file.log', "r")
vm_evt_file = open(evt_path+'/vm_event_file.log', "r")
met_file = open(met_path+'/metric_file.log', "r")

sampling_interval = int(args.sampling_interval)
process_id = int(args.process_id)
interval_count = int(args.interval_count)

print("sampling_interval: ", sampling_interval)
if process_id == None:
    print("[Warning] pid not given. collecting all system-wide data")
else:
    print("process id: ", process_id)

killall_perf = "killall perf"
evt_cmd = ""

''' from event_file '''
for evt in evt_file:
    evt_cmd = evt_cmd+evt.lstrip().rstrip()+","
# evt_cmd = evt_cmd[0:len(evt_cmd)-1]

''' from cache event file '''
for evt in cache_evt_file:
    evt_cmd = evt_cmd+evt.lstrip().rstrip()+","
# evt_cmd = evt_cmd[0:len(evt_cmd)-1]

''' float event file '''
for evt in float_evt_file:
    evt_cmd = evt_cmd+evt.lstrip().rstrip()+","
# evt_cmd = evt_cmd[0:len(evt_cmd)-1]

''' frontend event file '''
for evt in frontend_evt_file:
    evt_cmd = evt_cmd+evt.lstrip().rstrip()+","
# evt_cmd = evt_cmd[0:len(evt_cmd)-1]

''' memory event file '''
for evt in memory_evt_file:
    evt_cmd = evt_cmd+evt.lstrip().rstrip()+","
# evt_cmd = evt_cmd[0:len(evt_cmd)-1]

''' hardware interrupt event file '''
for evt in hwinter_evt_file:
    evt_cmd = evt_cmd+evt.lstrip().rstrip()+","
# evt_cmd = evt_cmd[0:len(evt_cmd)-1]

''' pipeline event file '''
for evt in pipeline_evt_file:
    evt_cmd = evt_cmd+evt.lstrip().rstrip()+","
# evt_cmd = evt_cmd[0:len(evt_cmd)-1]

''' uncore event file '''
for evt in uncore_evt_file:
    evt_cmd = evt_cmd+evt.lstrip().rstrip()+","
# evt_cmd = evt_cmd[0:len(evt_cmd)-1]

''' uncore event file '''
for evt in vm_evt_file:
    evt_cmd = evt_cmd+evt.lstrip().rstrip()+","
evt_cmd = evt_cmd[0:len(evt_cmd)-1]

fire_event = "perf stat --post /mnt/B/sem3/mss/project/post.sh -e {} -o /mnt/B/sem3/mss/project/temp1.log -I {} --interval-count {}".format( evt_cmd, sampling_interval, interval_count)

'''---------------------------------------------------------------------------------'''
''' metric file '''
'''---------------------------------------------------------------------------------'''

met_cmd = ""

for evt in met_file:
    met_cmd = met_cmd+evt.lstrip().rstrip()+","
met_cmd = met_cmd[0:len(met_cmd)-1]

'''---------------------------------------------------------------------------------'''

fire_metric = "perf stat --post /mnt/B/sem3/mss/project/post.sh -M {} -o /mnt/B/sem3/mss/project/temp2.log -I {} --interval-count {}".format( met_cmd, sampling_interval, interval_count)

log_file.write(fire_event)
log_file.write("\n\n")
log_file.write(fire_metric)

args1 = shlex.split(fire_event)
args2 = shlex.split(fire_metric)

proc1 = Process(target=fire_in_the_hole, args=(args1,))
proc2 = Process(target=fire_in_the_hole, args=(args2,))

proc1.start()
proc2.start()

proc1.join()
proc2.join()
# if process_id == None:
#     cmd = cmd + " -p {}".format(process_id)
# else:
#     cmd = cmd + " -a"