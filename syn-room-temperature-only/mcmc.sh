#!/usr/bin/env bash

# Make sure logging folders exist
LOG="./log/mcmc-syn"
mkdir -p $LOG

echo "## mcmc syn." >> log/save_pid.log

# for cell in $CELLS  # or
for ((x=0; x<25; x++));
do
	echo "Cell $x"
	nohup python mcmc.py $x > $LOG/cell_$x.log 2>&1 &
	echo "# cell $x" >> log/save_pid.log
	echo $! >> log/save_pid.log
	sleep 5
done

