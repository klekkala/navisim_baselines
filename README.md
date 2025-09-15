# PPO Training on ManiSkill (PickCube-v1)

This project trains a **Proximal Policy Optimization (PPO)** agent on the [ManiSkill](https://github.com/haosulab/ManiSkill) environment **PickCube-v1**, using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

---

## 📂 Structure
├── main.py     # Training + evaluation script
├── result/     # Saved GIFs and keyframes
├── logs/       # Training logs and checkpoints
├── videos/     # Training videos
---

## 🚀 Features
- Train PPO agent on **PickCube-v1**.
- Save checkpoints every **10k steps**.
- TensorBoard logging support.
- Save evaluation episodes as **GIFs** and **keyframes**.

---

## ⚙️ Setup
```bash
Clone this repository and create the environment from `ppo.yaml`:

```bash
conda env create -f ppo.yaml
conda activate maniskill-py39

