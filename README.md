# navisim_baselines
navisim_baselines
content = """# 🤖 Reinforcement Learning Training Suite: DDPG / SAC / TD-MPC2 (ManiSkill2)

This repository provides training scripts for reinforcement learning agents using [ManiSkill2](https://github.com/haosulab/ManiSkill2) environments. It supports:

- ✅ [DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
- ✅ [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- ✅ TD-MPC2 with multitask support and TensorBoard logging

---

## 📁 Project Structure

| File                   | Description                                                      |
|------------------------|------------------------------------------------------------------|
| `train_ddpg.py`        | Train DDPG on PickCube-v1, save model, GIFs, and keyframes       |
| `train_sac.py`         | Train SAC on PickCube-v1, save model, GIFs, and keyframes        |
| `train_tensorboard.py` | Train TD-MPC2 (single-task or multi-task), with full logging     |
| `README.md`            | Project documentation                                            |

---

## 🧩 Requirements

```bash
pip install stable-baselines3[extra] mani-skill matplotlib torch pillow
pip install hydra-core omegaconf termcolor
