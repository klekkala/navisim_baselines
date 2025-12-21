#!/bin/bash
# Training script with LoRA - Much lower memory usage
# This script can be run from anywhere

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIRFT_DIR="$PROJECT_ROOT/src/virft"

# Change to virft directory (required for training)
cd "$VIRFT_DIR" || { echo "Error: Cannot cd to $VIRFT_DIR"; exit 1; }

# Activate environment
source /lab/student/anaconda3/bin/activate Visual-RFT

# Set Python path
export PYTHONPATH="$VIRFT_DIR/src:$PYTHONPATH"

export DEBUG_MODE="true"
export LOG_PATH="./maniskill_grpo_lora_training.log"

# Data paths
export DATA_PATH="/lab/student/kristine/Visual-RFT/maniskill_virft/pickcube_virft_data/dataset.json"
export CKPT_PATH="/lab/student/kristine/Visual-RFT/share_models/Qwen2-VL-2B-Instruct"
export SAVE_PATH="/lab/student/kristine/Visual-RFT/output/Qwen2-VL-2B-ManiSkill-GRPO-LoRA"

# Check if dataset exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Dataset not found at $DATA_PATH"
    exit 1
fi

echo "========================================="
echo "ManiSkill GRPO Training with LoRA"
echo "========================================="
echo "Dataset: $DATA_PATH"
echo "Model: $CKPT_PATH"
echo "Save path: $SAVE_PATH"
echo "Using LoRA to reduce memory usage"
echo "========================================="

# Create output directory
mkdir -p $SAVE_PATH

# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0

# Run training with LoRA
torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_maniskill.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "all-linear" \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --gradient_checkpointing false \
    --attn_implementation sdpa \
    --max_pixels 100000 \
    --max_steps 20 \
    --run_name Qwen2-VL-2B-ManiSkill-GRPO-LoRA \
    --save_steps 10 \
    --save_only_model true \
    --num_generations 2

echo "========================================="
echo "Training completed!"
echo "Model saved to: $SAVE_PATH"
echo "========================================="
