#!/usr/bin/env bash

FILE_IDs=("herg25oc"
          "herg27oc"
          "herg30oc"
          "herg33oc"
          "herg37oc")

for i in "${FILE_IDs[@]}"
do
    echo "Plotting $i"
    python paper-fitting-and-validation-selected-cells-zoom.py $i
done
