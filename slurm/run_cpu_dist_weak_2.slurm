#!/bin/csh
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH -o $HOME/slurmlog/slurm-%j.out-%N
#SBATCH -e $HOME/slurmlog/slurm-%j.err-%N
#SBATCH --ntasks=1
#SBATCH --account=usucs5030
#SBATCH --partition=kingspeak
setenv WORKDIR $HOME/hpc_project/src/cpu_distributed
setenv SCRDIR /scratch/general/vast/$USER/$SLURM_JOBID
mkdir -p $SCRDIR
cp -r $WORKDIR/* $SCRDIR
cd $SCRDIR

module load intel impi

mpirun -n 1 ./cpu_distributed 2
