#!/bin/bash

# Set paths
JOB_DIR="./finetune_lr_5e-4"  # Output directory
CSV_PATH_TRAIN="csvs/dataset_multi_abnormality_labels_train_predicted_labels.csv"
CSV_PATH_VAL="csvs/dataset_multi_abnormality_labels_test_set.csv"
DATA_DIR="/mnt/nas/CT/npz_npy_valid/"  # Dataset directory
PRETRAIN_CHKPT="/home/than/DeepLearning/CMIL/checkpoints/checkpoint-799.pth"  # Path to pre-trained checkpoint


# Run the script
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 finetune_mae.py \
    --batch_size 1 \
    --accum_iter 128 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 50 \
    --warmup_epochs 5 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 \
    --output_dir ${JOB_DIR} \
    --log_dir ${JOB_DIR} \
    --num_workers 12 \
    --csv_path_train ${CSV_PATH_TRAIN} \
    --csv_path_val ${CSV_PATH_VAL} \
    --nb_classes 18 \
    --lmdb_path ${DATA_DIR} \
    --pin_mem \
    --smoothing 0.0 \
    --start_epoch 12 \
    --resume checkpoints/checkpoint-11.pth \
    --eval