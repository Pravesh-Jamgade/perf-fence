import os
import sys
import shlex, subprocess
import argparse
from subprocess import check_output, check_call, CalledProcessError
parser = argparse.ArgumentParser(
                    prog='PEMtool',
                    description='PMU Events Metric collector tool',
                    epilog='EPILOG')


parser.add_argument('-s', "--sampling_interval", help="sample period (ms)")
parser.add_argument('-p', "--process_id", help="process id, if not given then system wide data collection")
parser.add_argument('-c', "--interval_count", help="no. of interval's to collect data for")

args = parser.parse_args()

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

for evt in met_file:
    cmd = "perf stat -a --post /mnt/B/sem3/mss/project/post.sh -M {} -o /mnt/B/sem3/mss/project/temp.log -I {} --interval-count {}".format(evt.lstrip().rstrip(), sampling_interval, interval_count)

    # if process_id != None:
    #     cmd = cmd + " -p {}".format(process_id)
    # else:
    #     cmd = cmd + " -a"
    
    try:
        args = shlex.split(cmd)
        p = check_call(args,stderr=subprocess.STDOUT)
        # log_file.write(evt.lstrip().rstrip()+" ...pass\n")

    except CalledProcessError as e:
        log_file.write(str(evt).lstrip().rstrip()+".....exception\n")
        log_file.write(cmd+'\n')
        continue
    
    