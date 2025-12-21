# Behavior Cloning on ManiSkill (PushCube-v1)

This project implements **Behavior Cloning (BC)** for imitation learning on the ManiSkill environment **PushCube-v1**, adapted from the CORL implementation:  
https://github.com/corl-team/CORL/blob/main/algorithms/offline/any_percent_bc.py

---

## 🚀 Features

- Train a Behavior Cloning agent directly from **offline demonstrations**
- Support for **state-based** imitation learning
- Configurable `--max-episode-steps` to handle long demonstration trajectories
- Lightweight and easy to integrate with ManiSkill workflows

---

## ⚙️ Environment Setup (behavior-cloning-ms)

We recommend creating a dedicated Conda environment:

```bash
conda create -n behavior-cloning-ms python=3.9
conda activate behavior-cloning-ms
pip install -e .
```

This assumes the project directory contains a `setup.py` or `pyproject.toml`.

---

## 📂 Reference Documentation

Before running training, read the ManiSkill imitation learning guide:

👉 https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html

This document explains:

- How to download demonstration datasets  
- How to preprocess data  
- How to evaluate imitation learning models fairly  
- Common pitfalls and performance tips  

It is highly recommended for ensuring reproducible experiments.

---

## 🏋️ Training Behavior Cloning

We provide scripts to train BC from demonstration data.

> Some demonstration types (e.g., motion planning or teleoperation) take longer than the default environment horizon.  
> Behavior Cloning learns to imitate the **speed** of demonstrations, so set `--max-episode-steps` to around **2× the mean demonstration length**.

### Example: State-Based BC Training on PushCube-v1

```bash
python bc.py --env-id "PushCube-v1"   --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.cpu.h5   --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max-episode-steps 100   --total-iters 10000
```

Key arguments:

- `--env-id`: ManiSkill environment name  
- `--demo-path`: trajectory dataset (`.h5` file)  
- `--control-mode`: control mode (e.g., `pd_ee_delta_pos`)  
- `--sim-backend`: backend (`cpu` / `gpu`)  
- `--max-episode-steps`: prevents early cutoff of long demos  
- `--total-iters`: total training iterations  

---

## 📦 (Optional) RGB‑D Training

Support may vary depending on the camera setup.  
If desired, RGB‑D training follows this template:

```bash
python bc_rgbd.py --env-id "PushCube-v1"   --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.cpu.h5   --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max-episode-steps 100   --total-iters 10000
```

---

## 📖 References

If you use this baseline in your work, please cite the CORL Behavior Cloning implementation and the ManiSkill benchmark.

---

This README provides a clean and practical structure consistent with the PPO / DDPG / ACT styles above.
