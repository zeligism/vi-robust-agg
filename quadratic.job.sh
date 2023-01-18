#!/bin/bash
#SBATCH --job-name=virobagg-q
#SBATCH -q cpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=train-%j.log

source activate torch

srun python run_experiments.py --quadratic
