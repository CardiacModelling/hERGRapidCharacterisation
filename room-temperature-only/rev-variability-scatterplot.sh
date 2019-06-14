#!/usr/bin/env bash

# (.) turns grep return into array
# use grep with option -e (regexp) to remove '#' starting comments
FILES=('../qc/herg25oc1-staircaseramp-Rseal_before.txt'
	'../qc/herg25oc1-staircaseramp-Cm_before.txt'
	'../qc/herg25oc1-staircaseramp-Rseries_before.txt'
	'../qc/herg25oc1-staircaseramp-leak_before-g.txt'
	'../qc/herg25oc1-staircaseramp-leak_before-e.txt'
	'../qc/herg25oc1-staircaseramp-EK_all.txt')

SAVE=('rseal'
	'cm'
	'rseries'
	'gleak'
	'eleak'
	'ek')

for ((x=0; x<5; x++));
do
	echo "${FILES[x]} ${SAVE[x]}"
	nohup python rev-variability-scatterplot.py ${FILES[x]} ${SAVE[x]} &
	sleep 5
done

