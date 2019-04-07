#!/usr/bin/env bash

FILENAME="herg27oc1"
FILEIDX="1"

# Make sure logging folders exist
LOG="./log/fit-$FILENAME"
mkdir -p $LOG

# (.) turns grep return into array
# use grep with option -e (regexp) to remove '#' starting comments
CELLS=(`grep -v -e '^#.*' ./manualselection/manualselected-$FILENAME.txt`)

echo "## fit-$FILENAME" >> log/save_pid.log

# for cell in $CELLS  # or
for ((x=0; x<20; x++));
do
	echo "${CELLS[x]}"
	nohup python fit.py $FILEIDX ${CELLS[x]} 10 > $LOG/${CELLS[x]}.log 2>&1 &
	echo "# ${CELLS[x]}" >> log/save_pid.log
	echo $! >> log/save_pid.log
	sleep 5
done

