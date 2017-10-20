#!/bin/bash
# NEUROCOMPUTING
METHODS="cascade_hybridranking_hybridsfs_genetic cascade_hybridsfs_genetic cascade_hybridranking_genetic genetic"
CLASSIFIERS='elm(10)'
for ROUND in {01..10}; do 
bash run_experiment_te.bash "$METHODS" "$CLASSIFIERS" $ROUND 
done &

