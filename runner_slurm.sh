#!/bin/bash

TOPDIR=$1
FIRST=$2
LAST=$3

python runner.py --first_event ${FIRST} --last_event ${LAST} --output_file cheap_reco_${FIRST}.hdf5
