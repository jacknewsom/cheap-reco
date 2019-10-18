#!/bin/bash

RECO_OUTPUT=/project/projectdirs/dune/users/marshalc/jack_scripts/reco_output
module load python

python histograms.py --u 5 --d $RECO_OUTPUT
python histograms.py --l 5 --u 10 --d $RECO_OUTPUT
python histograms.py --l 10 --u 100 --d $RECO_OUTPUT
python histograms.py --l 100 --d $RECO_OUTPUT

