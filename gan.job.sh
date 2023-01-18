#!/bin/bash
#SBATCH --job-name=virobagg-g
#SBATCH -q gpu-single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=train-%j.log

source activate torch

python run_experiments.py --gan --use-cuda
