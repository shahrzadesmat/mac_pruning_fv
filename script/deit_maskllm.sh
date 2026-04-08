#!/bin/bash
#SBATCH --job-name=deit_maskllm_more_masks_agent_nm_1
#SBATCH --output=output/runs/deit_maskllm_more_masks_agent_nm_1.txt
#SBATCH --error=output/runs/deit_maskllm_error_more_masks_agent_nm_1.txt
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=bfxa-delta-gpu

# Load CUDA module
module use cuda/12.8

source /work/hdd/bfxa/dshah13/mac_pruning_fv/.venv/bin/activate

# export WANDB_SERVICE_WAIT=120
# export WANDB_START_METHOD=thread
export OPENROUTER_API_KEY=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
# export WANDB_API_KEY=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

MODEL_NAME="deit_tiny_distilled_patch16_224.fb_in1k"
MACS_TARGET_G="0.62"
DATASET="cifar10"

# CREATE ORGANIZED OUTPUT DIRECTORY STRUCTURE (updated to shahrzad directory)
BASE_OUTPUT_DIR="/work/hdd/bfxa/dshah13"
EXPERIMENT_DIR="${BASE_OUTPUT_DIR}/expf_cl/${MODEL_NAME}_${DATASET}_macs${MACS_TARGET_G/./p}g"
JOB_OUTPUT_DIR="${EXPERIMENT_DIR}/job_${SLURM_JOB_ID}"

# Create all necessary subdirectories
mkdir -p "$JOB_OUTPUT_DIR"
mkdir -p "${JOB_OUTPUT_DIR}/models"
mkdir -p "${JOB_OUTPUT_DIR}/logs"
mkdir -p "${JOB_OUTPUT_DIR}/checkpoints"

if [ $? -eq 0 ]; then
    echo "✅ Created experiment directory: $EXPERIMENT_DIR"
    echo "✅ Created job output directory: $JOB_OUTPUT_DIR"
else
    echo "❌ Failed to create output directories"
    exit 1
fi

# REDIRECT ALL OUTPUT TO ORGANIZED LOG FILE
LOG_FILE="${JOB_OUTPUT_DIR}/logs/pruning_workflow_${SLURM_JOB_ID}.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "🚀 STARTING PRUNING EXPERIMENT"
echo "========================================="
echo "Model: $MODEL_NAME"
echo "MACS Target: $MACS_TARGET_G"
echo "Dataset: $DATASET"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment Dir: $EXPERIMENT_DIR"
echo "Job Output Dir: $JOB_OUTPUT_DIR"
echo "Log File: $LOG_FILE"
echo "Final models will be saved to: ${JOB_OUTPUT_DIR}/models"
echo "Intermediate checkpoints will be saved to: ${JOB_OUTPUT_DIR}/checkpoints"
echo "========================================="

# Run the pruning workflow with organized output directories
echo "Starting ImageNet workflow..."
python3 main.py \
    --model "$MODEL_NAME" \
    --macs_target_g "$MACS_TARGET_G" \
    --dataset "$DATASET" \
    --accuracy_threshold 1.0 \
    --macs_undershoot_tolerance_pct 20.0 \
    --output_dir "${JOB_OUTPUT_DIR}/models" \
    --checkpoint_dir "${JOB_OUTPUT_DIR}/checkpoints" \
    --wandb_project "Isomorphic_Pruning_Experiments" \
    --wandb_mode offline \
    --wandb_name "${MODEL_NAME}_${DATASET}_macs${MACS_TARGET_G/./p}_job${SLURM_JOB_ID}" \
    --pruning_method maskllm

