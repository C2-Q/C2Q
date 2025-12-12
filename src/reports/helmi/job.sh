#!/bin/bash

#SBATCH --job-name=helmi_test   # Job name
#SBATCH --account=project_<id>  # Project for billing (slurm_job_account)
#SBATCH --partition=q_fiqci   # Partition (queue) name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --mem-per-cpu=1G       # memory allocation
#SBATCH --cpus-per-task=1     # Number of cores (threads)
#SBATCH --time=00:15:00         # Run time (hh:mm:ss)

module use /appl/local/quantum/modulefiles
module load helmi_qiskit

python helmi_test.py