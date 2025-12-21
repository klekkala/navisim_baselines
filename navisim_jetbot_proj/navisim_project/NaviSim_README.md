# NaviSim Navigation — Training & Visualization Pipeline

This repository provides a complete pipeline for training, evaluating, and visualizing navigation policies in **NaviSim** using Isaac Lab environments. The unified script `train_navisim_v2.py` supports data collection, RL training, testing, and visualization in a consistent workflow.

---

## 🚀 Features

- **Data Collection**  
  Generate trajectories using scripted baseline navigation policies.

- **Reinforcement Learning Training**  
  Supports **SAC** and **PPO** with replay buffers, gradient updates, checkpoints, and TensorBoard logging.

- **Policy Testing**  
  Run trained agents with **Jetbot first-person camera** and automatically save trajectories.

- **Visualization Tools**  
  Export:
  - GIFs of test episodes  
  - Key frame snapshots  
  - Episode statistics  
  - Recorded videos  

---

## 📂 Directory Structure

```
result_navisim/      # GIFs, key frames, evaluation outputs
logs_navisim/        # Training logs, checkpoints, TensorBoard data
videos_navisim/      # Test episode recordings
buffers_navisim/     # Replay buffers used for RL training
```

---

## 🧠 Usage

### 1️⃣ Collect baseline data
```bash
./run_isaac_lab.sh "python train_navisim_v2.py --mode collect --num_envs 4 --total_steps 50000"
```

### 2️⃣ Train a SAC policy
```bash
./run_isaac_lab.sh "python train_navisim_v2.py --mode train --policy_type sac --num_envs 4 --total_steps 100000"
```

### 3️⃣ Test a trained policy and generate visualizations
```bash
./run_isaac_lab.sh "python train_navisim_v2.py --mode test --policy_type sac --model_path logs_navisim/sac_policy_final.pth --num_episodes 5"
```

### 4️⃣ Train and evaluate in one run
```bash
./run_isaac_lab.sh "python train_navisim_v2.py --mode both --policy_type sac --num_envs 4 --total_steps 100000 --num_episodes 5"
```

---

## 📌 Notes

- Works with Isaac Lab environments configured for NaviSim.
- Use TensorBoard to monitor training:
  ```bash
  tensorboard --logdir logs_navisim
  ```
- Ensure GPU rendering is available when collecting camera observations.

