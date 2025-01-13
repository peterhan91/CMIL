#!/bin/bash

# Set paths
JOB_DIR="./finetune_ct_clip"  # Output directory
CSV_PATH_TRAIN="csvs/dataset_multi_abnormality_labels_test_set.csv"
CSV_PATH_VAL="csvs/dataset_multi_abnormality_labels_test_set.csv"
DATA_DIR="/mnt/nas/CT/npz_npy_valid/"  # Dataset directory
PRETRAIN_CHKPT="/home/than/DeepLearning/CMIL/checkpoints/merlin/i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"  # Path to pre-trained checkpoint


# Run the script
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 finetune_merlin.py \
    --batch_size 1 \
    --accum_iter 64 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 50 \
    --warmup_epochs 5 \
    --blr 5e-4 \
    --output_dir ${JOB_DIR} \
    --log_dir ${JOB_DIR} \
    --num_workers 12 \
    --csv_path_train ${CSV_PATH_TRAIN} \
    --csv_path_val ${CSV_PATH_VAL} \
    --nb_classes 18 \
    --lmdb_path ${DATA_DIR} \
    --pin_mem \
    --smoothing 0.0 \
