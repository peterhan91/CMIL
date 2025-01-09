#!/bin/bash
# ========================================
# Bash Script to Run MAE Main Function
# ========================================

#SBATCH -A p0021834
#SBATCH -p c23g
#SBATCH --job-name="finetune_mae"
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
JOB_DIR="/hpcwork/p0021834/workspace_tianyu/mae_runs/finetune"
CSV_PATH="/hpcwork/p0021834/workspace_tianyu/codes/CMIL/csvs/ct_rate_train_all.csv"  # Path to CSV file
DATA_DIR="/hpcwork/p0021834/workspace_tianyu/ct_rate/"  # Dataset directory
PRETRAIN_CHKPT="/hpcwork/p0021834/workspace_tianyu/mae_runs/checkpoint-799.pth"  # Path to pre-trained checkpoint


# Run the script
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune_mae.py \
    --batch_size 4 \
    --accum_iter 64 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 \
    --warmup_epochs 40 \
    --output_dir ${JOB_DIR} \
    --log_dir ${JOB_DIR} \
    --num_workers 24 \
    --csv_path ${CSV_PATH} \
    --nb_classes 18 \
    --lmdb_path ${DATA_DIR} \
    --pin_mem \

# ========================================
# End of Script
# ========================================