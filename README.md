# Intel Kaby Lake R 
1. sudo apt-get install linux-tools-6.2.0-32-generic\n
1. sudo echo -1 > /proc/sys/kernel/perf_event_paranoid
1. sudo echo 0 > /proc/sys/kernel/nmi_watchdog



----------------------------------------
perf stat --post /mnt/B/sem3/mss/project/post.sh -ebranch-misses,cache-misses,LLC-loads,node-stores -p 266791 -I 5000 --interval-count 2 -o /mnt/B/sem3/mss/project/temp.log

perf stat --post /mnt/B/sem3/mss/project/post.sh -ebranch-misses,cache-misses,LLC-loads,node-stores -p 266791 -I 5000 --interval-count 5


sudo echo -1 > /proc/sys/kernel/perf_event_paranoid
sudo echo 0 > /proc/sys/kernel/nmi_watchdog


naming style

condition[wi/wo].version[v1,v2..].metric/event[temp1/temp2].log
ex. without DDoS --> wo.v1.temp1.log and wo.v1.temp2.log

# usage
1. main.py:      collect metrics
2. read.py:      make sure no files other than generated from main.py present in o/p folder. output folder path, flag=0(no attack) It does all cleaning, concatening
3. gmm.py:      single gmm, run from output
4. batch_para_gmm.py: batched gmm 100 parameter per gmm model
5. online.py: daemon process collect, train (from output/all_df.log) and test (from folder_name/   as it generated temp files)


------------------------------------------------
all_df.log training samples on 11 Nov
a1 -> whole night no activity 1000 samples
a2 -> while normal usage 1000 samples
a3 -> 60 samples of attack

---------------
attack bin
for i in $(yes | sed 10q); do ./a.out & done
