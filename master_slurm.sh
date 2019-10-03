#!/bin/bash
#SBATCH -A dune
#SBATCH -L project,projecta 
#SBATCH -q shared
#SBATCH -C haswell
#SBATCH --image=deeplearnphysics/larcv2:cuda90-pytorch-dev20181015-scn
#SBATCH --job-name=dune-reco-pca

NPER=$1

TOPDIR="/global/project/projectdirs/dune/users/marshalc/jack_scripts/cheap-reco"

FIRST=$((${NPER}*${SLURM_ARRAY_TASK_ID}))
LAST=$((${FIRST}+${NPER}))

srun shifter --volume=/global/project:/project --volume=/global/projecta:/projecta ${TOPDIR}/runner_slurm.sh ${TOPDIR} ${FIRST} ${LAST}

