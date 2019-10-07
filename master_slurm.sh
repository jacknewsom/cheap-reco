#!/bin/bash
#SBATCH -A dune
#SBATCH -L project,projecta 
#SBATCH -q shared
#SBATCH -C haswell
#SBATCH --image=deeplearnphysics/larcv2:cuda90-pytorch-dev20181015-scn
#SBATCH --job-name=dune-reco-pca
#SBATCH -t 06:00:00

TOPDIR="/global/project/projectdirs/dune/users/marshalc/jack_scripts"
SEED=${SLURM_ARRAY_TASK_ID}
INFILE="bareroot_${SEED}"
OUTFILE="reco_output_${SEED}"

srun ${TOPDIR}/cheap-reco/runner_slurm.sh ${TOPDIR} ${INFILE} ${OUTFILE} ${SEED} 10
