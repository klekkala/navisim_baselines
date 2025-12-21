# PPO Training on ManiSkill (PickCube-v1)

This project trains a **Proximal Policy Optimization (PPO)** agent on the [ManiSkill](https://github.com/haosulab/ManiSkill) environment **PickCube-v1**, using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

---

## 🚀 Features
- Train PPO agent on **PickCube-v1**.
- Save checkpoints every **10k steps**.
- TensorBoard logging support.
- Save evaluation episodes as **GIFs** and **keyframes**.

---

## ⚙️ Setup

Clone this repository and create the environment from `ppo.yaml`:

```bash
conda env create -f ppo.yaml
conda activate ppo
```

---

## 🏋️ Training

Run PPO training:

```bash
python train_ppo.py
```

- Uses `MlpPolicy` with state observations.
- Trains for **1M timesteps**.
- Saves checkpoints in `logs/`.
- Final model: `logs/ppo_maniskill_pickcube_final.zip`.

---

## 📊 Monitoring

To visualize training progress with TensorBoard:

```bash
tensorboard --logdir logs/ppo_tensorboard
```

---

## 🎮 Evaluation

After training, the script will:
- Run **5 evaluation episodes**.
- Save GIFs to `result/episode_*.gif`.
- Save keyframes to `result/ep*_step*.png`.

---

## 📦 Dependencies

The Conda environment is defined in [ppo.yaml](ppo.yaml).  
Key packages:
- `torch==2.7.0`
- `stable-baselines3==2.6.0`
- `gymnasium==0.29.1`
- `mani-skill==3.0.0b20`
- `matplotlib`, `moviepy`, `pillow`, `pandas`

---

## 📖 References
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [ManiSkill Environments](https://github.com/haosulab/ManiSkill)
- [PPO Paper](https://arxiv.org/abs/1707.06347)