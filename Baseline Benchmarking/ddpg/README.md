# DDPG Training on ManiSkill (PickCube-v1)

This project trains a **Deep Deterministic Policy Gradient (DDPG)** agent on the [ManiSkill](https://github.com/haosulab/ManiSkill) environment **PickCube-v1**, using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

---

## 🚀 Features
- Train a DDPG agent on **PickCube-v1** (continuous control).
- Save model checkpoints every **10k steps**.
- TensorBoard logging support.
- Automatic recording of evaluation episodes as **GIFs** and **keyframes**.

---

## ⚙️ Setup

Create and activate the environment using `ddpg.yaml`:

```bash
conda env create -f ddpg.yaml
conda activate ddpg
```

---

## 🏋️ Training

Run DDPG training:

```bash
python train_ddpg.py
```

Training configuration:

- Uses `MlpPolicy` with **state observations**
- Control mode: `pd_ee_delta_pose`
- Dense reward shaping
- Total training steps: **1M**
- Checkpoints saved under `logs_ddpg/`
- Final model:  
  ```
  logs_ddpg/ddpg_maniskill_pickcube_final.zip
  ```

---

## 📊 Monitoring

Use TensorBoard to visualize critic loss, actor updates, learning curves, etc.:

```bash
tensorboard --logdir logs_ddpg/ddpg_tensorboard
```

---

## 🎮 Evaluation

After training completes, the script automatically evaluates the trained agent:

### Saved Outputs
- **GIFs**:  
  `result_ddpg/episode_*.gif`
- **Keyframes** (0%, 25%, 50%, 75%, 100% of each episode):  
  `result_ddpg/ep*_step*.png`

This allows quick visual debugging of grasping performance.

---

## 📦 Dependencies

Defined in **ddpg.yaml**.

Key packages include:

- `torch==2.7.0`
- `stable-baselines3==2.6.0`
- `gymnasium==0.29.1`
- `mani-skill==3.0.0b20`
- `matplotlib`
- `pillow`
- `moviepy` (optional for video manipulation)

---

## 📖 References
- [DDPG Paper](https://arxiv.org/abs/1509.02971)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [ManiSkill Environments](https://github.com/haosulab/ManiSkill)
