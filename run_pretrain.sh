#!/bin/bash

# ========================================
# Bash Script to Run MAE Main Function
# ========================================

# Set job directory, data path, and pretrained model path
# Replace these paths with your actual directories
JOB_DIR="runs/"
CSV_PATH="csvs/ct_rate_train_512.csv"  # Path to the CSV file containing the dataset information
DATA_DIR="/mnt/nas/Datasets/than/CT/LMDB/ct_rate_train_512.lmdb"  # Directory containing your dataset

# Run the main training script
torchrun --nproc_per_node=3 pretrain_mae.py \
    --batch_size 4 \
    --epochs 800 \
    --model mae_vit_base_patch32 \
    --mask_ratio 0.9 \
    --norm_pix_loss \
    --weight_decay 0.05 \
    --blr 1.6e-3 \
    --warmup_epochs 0 \
    --output_dir ${JOB_DIR} \
    --log_dir ${JOB_DIR} \
    --num_workers 12 \
    --csv_path ${CSV_PATH} \
    --lmdb_path ${DATA_DIR} \


# ========================================
# End of Script
# ========================================
