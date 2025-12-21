# Visual-RFT GRPO Training for ManiSkill

Training Qwen2-VL vision-language model for robot control using GRPO with LoRA.

This repository contains training configurations and scripts for using Visual-RFT with ManiSkill environments.

---

## 🚀 Quick Start

### 1. Download Visual-RFT Framework

Clone the official Visual-RFT repository:

```bash
# Navigate to your working directory
cd /your/workspace

# Clone Visual-RFT framework
git clone https://github.com/ORIGINAL_AUTHOR/Visual-RFT.git
cd Visual-RFT
```

**What is Visual-RFT?**
- Vision-language model fine-tuning framework for robotics
- Uses GRPO (Group Relative Policy Optimization)
- Trains vision-language models to predict robot actions

---

### 2. Download This ManiSkill Configuration

Clone this repository into the Visual-RFT directory:

```bash
# Inside Visual-RFT directory
# Clone the navisim_baselines repository
git clone https://github.com/klekkala/navisim_baselines.git
cd navisim_baselines/Novel-rl-algorithms/Visual-RFT

# The ManiSkill → Visual-RFT pipeline is located here
cd maniskill_virft
```

**Directory structure:**
```
Visual-RFT/
├── src/virft/              # Visual-RFT framework code
├── maniskill_virft/        # This repository (ManiSkill configs)
│   ├── environment.yml
│   ├── train_grpo_lora.sh
│   ├── pickcube_virft_data/
│   └── README.md
```

---

### 3. Create Environment

```bash
# From Visual-RFT/maniskill_virft directory
cd maniskill_virft
conda env create -f environment.yml
conda activate Visual-RFT
```

**Environment includes:**
- Python 3.10
- PyTorch 2.7.1 with CUDA 12
- Transformers 4.53.2
- TRL 0.19.1 (for GRPO)
- PEFT 0.18.0 (for LoRA)
- ManiSkill 3.0.0b21
- Other dependencies

**Note:** DeepSpeed removed. LoRA is used for memory efficiency.

---

### 4. Download Qwen2-VL Model

Download the pre-trained Qwen2-VL-2B-Instruct model:

```bash
# Option 1: Using HuggingFace CLI (recommended)
cd /your/workspace/Visual-RFT
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir ./share_models/Qwen2-VL-2B-Instruct

# Option 2: Using git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct ./share_models/Qwen2-VL-2B-Instruct
```

Or update the model path in `train_grpo_lora.sh`:
```bash
export CKPT_PATH="/path/to/your/Qwen2-VL-2B-Instruct"
```

---

### 5. Run Training

```bash
# From anywhere (script auto-navigates)
cd /your/workspace/Visual-RFT/maniskill_virft
bash train_grpo_lora.sh
```

**Training Configuration:**
- Model: Qwen2-VL-2B-Instruct with LoRA
- Dataset: 7,720 samples (included)
- Steps: 20 (default, increase for full training)
- Memory: ~18GB (fits RTX 3090 Ti 24GB)
- Training time: ~40 seconds for 20 steps

---

## 📦 What's Included

This repository contains:

- `environment.yml` - Conda environment configuration
- `train_grpo_lora.sh` - LoRA training script
- `pickcube_virft_data/` - Pre-prepared ManiSkill PickCube dataset (7,720 samples)
- `convert_maniskill_to_virft.py` - Dataset conversion tool
- `grpo_maniskill.py` - Training code backup
- `README.md` - This file

---

## ⚙️ Training Configuration

Edit `train_grpo_lora.sh` to customize:

```bash
# Paths (update these to match your setup)
export DATA_PATH="pickcube_virft_data/dataset.json"
export CKPT_PATH="../share_models/Qwen2-VL-2B-Instruct"
export SAVE_PATH="../output/Qwen2-VL-2B-ManiSkill-GRPO-LoRA"

# Training steps
--max_steps 20              # Increase to 1000+ for full training

# LoRA parameters
--lora_r 16                 # LoRA rank
--lora_alpha 32             # LoRA alpha
--lora_dropout 0.05

# Training hyperparameters
--per_device_train_batch_size 1
--gradient_accumulation_steps 2
--learning_rate 1e-6        # Can increase to 1e-5
```

**Important:**
- `--gradient_checkpointing false` must be kept (conflicts with LoRA+DDP)
- `--use_peft true` enables LoRA and reduces memory usage

---

## 🔧 Why LoRA?

**Problem:**
- GRPO needs 2 model copies (policy + reference) = ~34GB
- RTX 3090 Ti only has 24GB VRAM

**Solution:**
- With LoRA, reference model is disabled
- Only 1 model copy needed = ~18GB
- Training is stable and faster

---

## 📊 Expected Output

```
=========================================
ManiSkill GRPO Training with LoRA
=========================================
Dataset: .../pickcube_virft_data/dataset.json
Model: .../Qwen2-VL-2B-Instruct
Using LoRA to reduce memory usage
=========================================

Using trainer: Qwen2VLGRPOTrainer
Starting training...
  5%|▌  | 1/20 [00:02<00:45, 2.38s/it]
100%|██████████| 20/20 [00:42<00:00, 2.14s/it]

Saving model to: .../output/Qwen2-VL-2B-ManiSkill-GRPO-LoRA
Training complete!
```

**Note:** Initial rewards are 0.0 - normal! Model needs 1000+ steps to learn.

---

## 🛠️ Troubleshooting

### Visual-RFT Not Found

```
Error: Cannot cd to /path/to/Visual-RFT/src/virft
```

**Solution:** Make sure you cloned Visual-RFT framework first
```bash
git clone https://github.com/ORIGINAL_AUTHOR/Visual-RFT.git
```

### Model Not Found

```
OSError: Model does not exist
```

**Solution:** Download Qwen2-VL or update path in `train_grpo_lora.sh`
```bash
export CKPT_PATH="/your/path/to/Qwen2-VL-2B-Instruct"
```

### CUDA Out of Memory

```
torch.OutOfMemoryError: CUDA out of memory
```

**Solution:** Reduce image resolution
```bash
--max_pixels 50000  # Instead of 100000
```

### PEFT Not Found

```
ValueError: PEFT library not installed
```

**Solution:** Activate environment
```bash
conda activate Visual-RFT
pip list | grep peft  # Should show peft 0.18.0
```

---

## ✅ Complete Setup Example

```bash
# 1. Clone Visual-RFT framework
git clone https://github.com/ORIGINAL_AUTHOR/Visual-RFT.git
cd Visual-RFT

# 2. Clone this ManiSkill configuration
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git maniskill_virft

# 3. Create environment
cd maniskill_virft
conda env create -f environment.yml
conda activate Visual-RFT

# 4. Download Qwen2-VL model
cd ..
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir ./share_models/Qwen2-VL-2B-Instruct

# 5. Start training
cd maniskill_virft
bash train_grpo_lora.sh

# 6. Monitor training
tail -f ../src/virft/maniskill_grpo_lora_training.log
```

**Total setup time:** 15-20 minutes (depends on download speed)

---

## 📌 Important Notes

- **Requires Visual-RFT framework** - This repo only contains ManiSkill configs
- **Training script** auto-navigates to `Visual-RFT/src/virft/`
- **Model checkpoints** saved to `Visual-RFT/output/`
- **Dataset included** - 7,720 pre-prepared PickCube samples
- **GPU requirement:** 24GB VRAM (RTX 3090/4090)

---

## 📁 Repository Structure

```
maniskill_virft/              # This repository
├── environment.yml           # Conda environment
├── train_grpo_lora.sh       # Training script
├── pickcube_virft_data/     # Dataset (7,720 samples)
│   ├── dataset.json
│   └── images/
├── convert_maniskill_to_virft.py
├── grpo_maniskill.py        # Training code backup
└── README.md                # This file
```

**Must be placed in Visual-RFT directory:**
```
Visual-RFT/                  # Main framework (download separately)
├── src/virft/               # Framework code
├── maniskill_virft/         # This repository
└── share_models/            # Download Qwen2-VL here
```

---

## 🎯 Next Steps After Training

1. **Increase training steps** to 1000+ for better performance
2. **Evaluate the model** on ManiSkill test episodes
3. **Fine-tune hyperparameters** (learning rate, LoRA rank)
4. **Deploy** trained LoRA adapter for robot control

---

## 📚 References

- **Visual-RFT:** Visual Reinforcement Fine-Tuning framework
- **GRPO:** Group Relative Policy Optimization
- **Qwen2-VL:** Vision-Language Model from Alibaba
- **LoRA:** Low-Rank Adaptation for efficient fine-tuning
- **ManiSkill:** GPU-parallelized robot manipulation benchmark
