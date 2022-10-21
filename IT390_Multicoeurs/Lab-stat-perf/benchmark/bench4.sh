#!/bin/sh
export OMP_NUM_THREADS=4
for i in `seq 1 50` ; do /usr/bin/time -f "%e" -o tmpTime ./matrix ; cat tmpTime >> bench2.txt ; done