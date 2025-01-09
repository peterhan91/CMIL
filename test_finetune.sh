#!/bin/bash

export WANDB_API_KEY=39c0441c544226b0f830737701c965ff9d068d41

# Set paths
JOB_DIR="/hpcwork/p0021834/workspace_tianyu/mae_runs/finetune"
CSV_PATH_TRAIN="/hpcwork/p0021834/workspace_tianyu/codes/CMIL/csvs/dataset_multi_abnormality_labels_train_predicted_labels.csv"
CSV_PATH_VAL="/hpcwork/p0021834/workspace_tianyu/codes/CMIL/csvs/dataset_multi_abnormality_labels_validation_set.csv"
DATA_DIR="/hpcwork/p0021834/workspace_tianyu/ct_rate/"  # Dataset directory
PRETRAIN_CHKPT="/hpcwork/p0021834/workspace_tianyu/mae_runs/checkpoint-799.pth"  # Path to pre-trained checkpoint


# Run the script
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 finetune_mae.py \
    --batch_size 1 \
    --accum_iter 1 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --warmup_epochs 0 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 \
    --output_dir ${JOB_DIR} \
    --log_dir ${JOB_DIR} \
    --num_workers 2 \
    --csv_path_train ${CSV_PATH_TRAIN} \
    --csv_path_val ${CSV_PATH_VAL} \
    --nb_classes 18 \
    --lmdb_path ${DATA_DIR} \
    --pin_mem \