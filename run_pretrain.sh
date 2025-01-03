#!/bin/bash
# ========================================
# Bash Script to Run MAE Main Function
# ========================================

#SBATCH -A p0021834
#SBATCH -p c23g
#SBATCH --job-name="pretrain_mae"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=374G
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
JOB_DIR="/hpcwork/p0021834/workspace_tianyu/mae_runs/"
CSV_PATH="/hpcwork/p0021834/workspace_tianyu/codes/CMIL/csvs/ct_rate_train_all.csv"  # Path to CSV file
DATA_DIR="/hpcwork/p0021834/workspace_tianyu/ct_rate/"  # Dataset directory

# Run the script
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 pretrain_mae.py \
    --batch_size 32 \
    --accum_iter 32 \
    --epochs 800 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.9 \
    --norm_pix_loss \
    --weight_decay 0.05 \
    --blr 1.5e-4 \
    --warmup_epochs 40 \
    --output_dir ${JOB_DIR} \
    --log_dir ${JOB_DIR} \
    --num_workers 24 \
    --csv_path ${CSV_PATH} \
    --lmdb_path ${DATA_DIR} \
    --start_epoch 580 \
    --resume /hpcwork/p0021834/workspace_tianyu/mae_runs/checkpoint-580.pth

# ========================================
# End of Script
# ========================================
