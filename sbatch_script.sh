#!/bin/bash
#SBATCH --job-name=byol_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=30G
#SBATCH --mem-per-gpu=18G

python -m byol_embeddings --path_to_dataset ../../processed_data/2020-10-28-AIS-005/ \
    -l outputs/wide101.pt --no_train