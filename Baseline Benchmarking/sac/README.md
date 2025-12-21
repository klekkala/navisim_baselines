# SAC Training on ManiSkill (PickCube-v1)

This project trains a **Soft Actor-Critic (SAC)** agent on the [ManiSkill](https://github.com/haosulab/ManiSkill) environment **PickCube-v1**.  
It uses [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for reinforcement learning.

The training script supports logging, checkpointing, TensorBoard visualization, and video recording.

---

## 📦 Environment Setup

We provide a full Conda environment export: [`sac.yaml`](./sac.yaml).

### 1. Create and activate the environment
```bash
conda env create -f sac.yaml
conda activate maniskill-sac
```

⚠️ **Note**:  
- If your system uses a different Conda installation path, please remove the last line in `sac.yaml` that begins with `prefix:` before creating the environment.  
- The provided `sac.yaml` specifies **CUDA 12.x**. If your GPU only supports CUDA 11, you may need to adjust the PyTorch version manually.

### 2. Verify installation
```bash
python -c "import torch, gymnasium, mani_skill, stable_baselines3; print(torch.__version__)"
```

Expected versions (from `sac.yaml`):
- `torch==2.7.0`  
- `mani-skill==3.0.0b20`  
- `stable-baselines3==2.6.0`  
- `gymnasium==0.29.1`  

---

## 🚀 Training

Run the training script:

```bash
python train_sac.py
```

The script will:
- Train an SAC agent on **PickCube-v1** with dense rewards.
- Save checkpoints in `logs_sac/`.
- Save the final model as `logs_sac/sac_maniskill_pickcube_final.zip`.
- Log TensorBoard data in `logs_sac/sac_tensorboard/`.

You can monitor training with TensorBoard:

```bash
tensorboard --logdir logs_sac/sac_tensorboard
```

---

## 🎥 Evaluation & Recording

After training, the script will run evaluation episodes:

- GIFs are saved in `result_sac/` (e.g., `episode_0.gif`).  
- Key frames are saved in `result_sac/` (e.g., `ep0_step50.png`).  
- Raw videos are recorded in `videos_sac/`.  

---

## 📂 Project Structure

```
.
├── sac.yaml                  # Full conda environment (exported from server, env name: maniskill-sac)
├── train_sac.py              # Training & evaluation script
├── logs_sac/                 # Checkpoints and TensorBoard logs
├── videos_sac/               # Raw evaluation videos
└── result_sac/               # GIFs and key frames
```

---

## ⚙️ Key Parameters

- Environment: `PickCube-v1`  
- Algorithm: `SAC`  
- Total Timesteps: `1,000,000`  
- Learning Rate: `3e-4`  
- Batch Size: `256`  
- Gamma: `0.95`  
- Replay Buffer: `1,000,000`  

You can adjust these hyperparameters directly in `train_sac.py`.

---

## ✅ Example Output

After training, you should see outputs such as:

```
result_sac/
├── episode_0.gif
├── episode_1.gif
├── ep0_step0.png
└── ep0_step50.png
```

---

## 📌 Notes

- Press `Ctrl + C` to safely stop training. Checkpoints are saved in `logs_sac/`.  
- The environment is heavy (full export), so installation may take time.  
- If CUDA-related packages fail to install, adjust the PyTorch + CUDA version according to your system.  