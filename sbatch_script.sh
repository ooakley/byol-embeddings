#!/bin/bash
#SBATCH --job-name=byol_train
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=128G
#SBATCH --mem-per-gpu=32G

python -m byol_embeddings \
    --path_to_dataset ../../processed_data/2021-02-25-AIS-006/dataset.npy \
    --path_to_labels ../../processed_data/2021-02-25-AIS-006/label_array.npy \
    --path_to_label_dictionary ../../processed_data/2021-02-25-AIS-006/label_dictionary.json \
    --ctrl_labels F91SCTRL L4QCTRL R45HCTRL WTCTRL \
    --path_to_model outputs/04-16-2021_15-05-23/model.pt \
    --no_train

# python -m byol_embeddings \
#     --path_to_dataset ../../processed_data/2019-08-21/dataset.npy \
#     --path_to_labels ../../processed_data/2019-08-21/label_array.npy \
#     --path_to_label_dictionary ../../processed_data/2019-08-21/label_dictionary.json \
#     --ctrl_labels WTCTRL \
#     -e 200
