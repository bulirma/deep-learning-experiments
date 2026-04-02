#!/bin/bash
#SBATCH --partition=mpi-homo-short      # partition you want to run job in
#SBATCH -n 1  # one job
#SBATCH -c 1  # one core
#SBATCH --output=build.log # stdout and stderr output file
#SBATCH -A nprg042s
#SBATCH --job-name="tbb-kmeans-build"

make
echo "Done."
