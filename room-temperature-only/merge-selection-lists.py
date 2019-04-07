#!/usr/bin/env python2

list1file = './manualselected-herg25oc1.txt'
list2file = './figs/manual-unselected-validation-herg25oc1.txt'
list1IsSelect = True
list2IsSelect = False

saveas = './manualv2selected-herg25oc1.txt'
description = "Based on manual selection of staircase-ramp (with auto QC)" \
              + " and validation recordings."

if not list1IsSelect:
    raise ValueError('List 1 provided must be a selection list')

# Read lists
list1 = []
with open(list1file, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            list1.append(l.split()[0])

list2 = []
with open(list2file, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            list2.append(l.split()[0])

# Start merging
if list2IsSelect:
    for i in list2:
        if i not in list1:
            list1.append(i)
    list1.sort()
else:
    for i in list2:
        if i in list1:
            list1.remove(i)

# Save merged list
with open(saveas, 'w') as f:
    f.write('# ' + description + '\n')
    for i in list1:
        f.write(i + '\n')
