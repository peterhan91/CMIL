#!/bin/bash

# ========================================
# Bash Script to Run MAE Main Function
# ========================================

# Set job directory, data path, and pretrained model path
# Replace these paths with your actual directories
JOB_DIR="runs"
CSV_PATH="csvs/ct_rate_train_512.csv"  # Path to the CSV file containing the dataset information
DATA_DIR="/home/than/Datasets/CT/ct_rate_train_512.lmdb"  # Directory containing your dataset

# Run the main training script
CUDA_VISIBLE_DEVICES=2,1 torchrun --nproc_per_node=2 pretrain_mae.py \
    --batch_size 32 \
    --accum_iter 64 \
    --epochs 800 \
    --model mae_vit_small_patch16 \
    --mask_ratio 0.9 \
    --norm_pix_loss \
    --weight_decay 0.05 \
    --blr 1.5e-4 \
    --warmup_epochs 40 \
    --output_dir ${JOB_DIR} \
    --log_dir ${JOB_DIR} \
    --num_workers 8 \
    --csv_path ${CSV_PATH} \
    --lmdb_path ${DATA_DIR} \
    --start_epoch 140 \
    --resume ${JOB_DIR}/checkpoint-140.pth


# ========================================
# End of Script
# ========================================
