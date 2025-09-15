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

⚙️ Environment

The Conda environment is fully specified in ppo.yaml.
Key dependencies include:
	•	torch==2.7.0
	•	stable-baselines3==2.6.0
	•	gymnasium==0.29.1
	•	mani-skill==3.0.0b20
	•	matplotlib, pandas, moviepy, pillow
