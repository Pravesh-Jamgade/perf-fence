## Kaby Lake R
perf stat --post /mnt/B/sem3/mss/project/post.sh -ebranch-misses,cache-misses,LLC-loads,node-stores -p 266791 -I 5000 --interval-count 2 -o /mnt/B/sem3/mss/project/temp.log

perf stat -a --topdown -p 266791

perf stat --post /mnt/B/sem3/mss/project/post.sh -ebranch-misses,cache-misses,LLC-loads,node-stores -p 266791 -I 5000 --interval-count 5


sudo echo -1 > /proc/sys/kernel/perf_event_paranoid
sudo echo 0 > /proc/sys/kernel/nmi_watchdog
