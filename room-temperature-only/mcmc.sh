#!/usr/bin/env bash

# Make sure logging folders exist
LOG="./log/mcmc-herg25oc1"
mkdir -p $LOG

# (.) turns grep return into array
# use grep with option -e (regexp) to remove '#' starting comments
CELLS=(`grep -v -e '^#.*' manualv2selected-herg25oc1.txt`)

echo "## mcmc-herg25oc1" >> log/save_pid.log

# for cell in $CELLS  # or
for ((x=0; x<20; x++));
do
	echo "${CELLS[x]}"
	nohup python mcmc.py ${CELLS[x]} > $LOG/${CELLS[x]}.log 2>&1 &
	echo "# ${CELLS[x]}" >> log/save_pid.log
	echo $! >> log/save_pid.log
	sleep 5
done

