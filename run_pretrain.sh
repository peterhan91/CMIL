#!/bin/bash

# ========================================
# Bash Script to Run MAE Main Function
# ========================================

# Set job directory, data path, and pretrained model path
# Replace these paths with your actual directories
OUTPUT_DIR="runs/"
CSV_PATH="csvs/ct_rate_train_512.csv"  # Path to the CSV file containing the dataset information
LMDB_PATH="/home/than/Datasets/CT/ct_rate_train_512.lmdb"  # Directory containing your dataset

# Run the main training script
python pretrain_mae_pl.py \
    --batch_size 32 \
    --epochs 400 \
    --accum_iter 64 \
    --model mae_vit_small_patch16 \
    --mask_ratio 0.9 \
    --norm_pix_loss \
    --weight_decay 0.05 \
    --blr 1.6e-3 \
    --warmup_epochs 40 \
    --csv_path $CSV_PATH \
    --lmdb_path $LMDB_PATH \
    --output_dir $OUTPUT_DIR \
    --log_dir $OUTPUT_DIR \
    --device cuda \
    --seed 0 \
    --num_workers 10 \
    --pin_mem \
    --use_amp \
    --print_freq 20


# ========================================
# End of Script
# ========================================
