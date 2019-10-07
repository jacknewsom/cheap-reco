#!/bin/bash

TOPDIR=$1
INFILE=$2
OUTFILE=$3
SEED=$4
NEVENTS=$5

module load python
python ${TOPDIR}/runner.py --input_file ${TOPDIR}/hdf5files/${INFILE}.hdf5 --output_file ${TOPDIR}/reco_output/${OUTFILE}.hdf5 --seed ${SEED} -n ${NEVENTS}
