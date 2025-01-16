#!/bin/bash
# ========================================
# Bash Script to Run MAE Main Function
# ========================================

#SBATCH -A p0021834
#SBATCH -p c23g
#SBATCH --job-name="finetune_mae_respect"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=tianyu.han@rwth-aachen.de

export WANDB_API_KEY=39c0441c544226b0f830737701c965ff9d068d41
module load CUDA/12.1.1

# Activate Conda environment
source /home/th900468/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate longvit

# Print debug information
echo "Environment Variables:"; export | grep -E "CUDA|CONDA"; echo
echo "NVIDIA-SMI Output:"; nvidia-smi; echo

# Set paths
JOB_DIR="/hpcwork/p0021834/workspace_tianyu/mae_runs/finetune_respect"
CSV_PATH_TRAIN="/hpcwork/p0021834/workspace_tianyu/codes/CMIL/csvs/rsna_2020_train.csv"
CSV_PATH_VAL="/hpcwork/p0021834/workspace_tianyu/codes/CMIL/csvs/rsna_2020_validation_set.csv"
DATA_DIR="/hpcwork/p0021834/workspace_tianyu/respect/"  # Dataset directory
PRETRAIN_CHKPT="/hpcwork/p0021834/workspace_tianyu/mae_runs/finetune_lr_5e-4/checkpoint-20.pth"  # Path to pre-trained checkpoint


# Run the script
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune_mae_inspect.py \
    --wandb_project_name "finetune_MAE_respect" \
    --task respect \
    --batch_size 2 \
    --accum_iter 128 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 50 \
    --warmup_epochs 5 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 \
    --output_dir ${JOB_DIR} \
    --log_dir ${JOB_DIR} \
    --num_workers 8 \
    --csv_path_train ${CSV_PATH_TRAIN} \
    --csv_path_val ${CSV_PATH_VAL} \
    --nb_classes 10 \
    --lmdb_path ${DATA_DIR} \
    --pin_mem \
    --smoothing 0.0 \


# ========================================
# End of Script
# ========================================