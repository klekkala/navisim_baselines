# Multi-Modal SAC for ManiSkill PickCube

Train a **Soft Actor-Critic (SAC)** agent with **R3M visual embeddings** and **proprioception** for ManiSkill **PickCube-v1**.

---

## 🚀 Quick Start

### Step 1: Clone and Setup

```bash
# Clone this repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Create conda environment
conda env create -f environment.yml
conda activate r3m_sac
```

### Step 2: Install R3M

```bash
# Clone R3M (visual representation library)
git clone https://github.com/facebookresearch/r3m.git
cd r3m
pip install -e .
cd ..
```

### Step 3: Run Training

```bash
python train_sac_r3m.py
```

That's it! Training will start immediately.

---

## 📦 What's Included

- `train_sac_r3m.py` - Complete SAC training script (self-contained)
- `environment.yml` - Conda environment with all dependencies
- `README.md` - This file

---

## 📊 Expected Output

```
=== Environment Info ===
Action dimension: 8
Visual feature dimension: 2048
Proprioception feature dimension: 28

=== Start Training ===
Ep    0 | Reward:   -5.23 | Success: 0.00% | Alpha: 0.368 | Buffer:    200
Ep   10 | Reward:   -2.89 | Success: 5.00% | Alpha: 0.341 | Buffer:   2200
Ep   50 | Reward:   -1.45 | Success: 15.00% | Alpha: 0.298 | Buffer:  10200
...
```

**Checkpoints saved:**
- `maniskill_r3m_best.pt` - Best model
- `maniskill_r3m_final.pt` - Final model

**Training time:** ~2-4 hours (1000 episodes on RTX 3090)

---

## 🔧 How It Works

### R3M Visual Encoder
- Pretrained ResNet-50 on Ego4D videos
- Converts RGB images (224×224) → 2048-D embeddings
- Frozen during training

### Proprioception
- Extracts robot state: joint positions, velocities, TCP pose
- From ManiSkill observations: `obs['agent']`, `obs['extra']`

### Multi-Modal SAC
- **Actor**: Vision MLP + Proprio MLP → Fusion → Policy
- **Critic**: Twin Q-networks for stability
- **Adaptive α**: Automatic entropy tuning
- **Replay Buffer**: 200K transitions
- **Reward Normalization**: Return-based scaling

---

## 📝 Key Hyperparameters

```python
gamma = 0.99                # Discount factor
batch_size = 256           # Training batch size
start_steps = 10000        # Random exploration
actor_lr = 1e-4            # Actor learning rate
critic_lr = 3e-4           # Critic learning rate
```

Edit these in `train_sac_r3m.py` if needed.

---

## 🛠️ Troubleshooting

**CUDA out of memory?**
- Reduce `batch_size = 128`
- Reduce `buffer_size = 100000`

**Low success rate?**
- This is expected! PickCube-v1 is challenging
- 20-30% success after 1000 episodes is good
- Try training longer (2000+ episodes)

**R3M installation fails?**
```bash
# Make sure you have git and pip
git clone https://github.com/facebookresearch/r3m.git
cd r3m
pip install -e .
```

---

## 📚 References

- **R3M**: https://arxiv.org/abs/2203.12601
- **SAC**: https://arxiv.org/abs/1801.01290
- **ManiSkill**: https://github.com/haosulab/ManiSkill

---

## 🎯 GPU Requirements

- **Minimum**: 8GB VRAM (RTX 3070)
- **Recommended**: 24GB VRAM (RTX 3090/4090)
- CPU training not supported (R3M requires GPU)

---

## 📄 License

MIT License (see LICENSE file)
