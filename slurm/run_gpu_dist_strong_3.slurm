#!/bin/csh
#SBATCH --time=1:00:00
#SBATCH --nodes=3
#SBATCH -o $HOME/slurmlog/slurm-%j.out-%N
#SBATCH -e $HOME/slurmlog/slurm-%j.err-%N
#SBATCH --ntasks=3
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu:v100
setenv WORKDIR $HOME/hpc_project/src/gpu_distributed
setenv SCRDIR /scratch/general/vast/$USER/$SLURM_JOBID
mkdir -p $SCRDIR
cp -r $WORKDIR/* $SCRDIR
cd $SCRDIR

module load intel impi
module load cuda/12.1

mpirun -np 3 ./a.out 10

