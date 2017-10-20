#!/bin/bash
if [ $# -lt 4 ]; then
EXPERIMENT=experiment_fs_te
else
EXPERIMENT=$4
fi
HOST=`hostname`
ROUND=$3
EXTENSION="txt$ROUND"
METHODS=$1
CLASSIFIERS=$2
for CLASSIFIER in $CLASSIFIERS; do
  for M in $METHODS; do
    OUTPUTFILE="results/neurocom/${HOST}.${EXPERIMENT}.${CLASSIFIER}.${M}.${EXTENSION}"; echo $OUTPUTFILE;
    METHOD=feature_selection_$M
    echo "bash matlab.bash <<< \"$EXPERIMENT('$METHOD', '$CLASSIFIER', $ROUND), quit\" > $OUTPUTFILE" > $OUTPUTFILE
    bash matlab.bash <<< "$EXPERIMENT('$METHOD', '$CLASSIFIER', $ROUND), quit" >> $OUTPUTFILE 
  done
done

