#!/bin/bash
#SBATCH --partition=mpi-homo-short      # partition you want to run job in
#SBATCH -n 1  # one job
#SBATCH -c 32  # cores
#SBATCH --output=run.log # stdout and stderr output file
#SBATCH --exclusive
#SBATCH -A nprg042s
#SBATCH --job-name="tbb-kmeans-run"

./k-means $@
