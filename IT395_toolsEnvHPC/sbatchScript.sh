#!/bin/bash -l

#SBATCH --job-name=runPerf
#SBATCH --partition=routage
# number of nodes
#SBATCH --nodes=1
# number of MPI process
#SBATCH --ntasks=1
# number of cores by process
#SBATCH --cpus-per-task=24
#SBATCH --exclusive

#SBATCH --time=1-00:00:00

mpirun ./prog

exit 0
