# Summer ManiSkill Research Portfolio

> Focus: RL Algorithms, Pretrained Representations, and Real-World Deployment

---

## Overview

This repository documents my comprehensive research on reinforcement learning (RL) for robotic manipulation tasks, conducted over this period. The work spans from baseline algorithm benchmarking to novel algorithm exploration, integration with pretrained visual representations, and real-world robot deployment.

### Research Highlights
- **Baseline Benchmarking**: Systematic evaluation of PPO, SAC, DDPG, TD-MPC2, ACT, and Behavior Cloning on ManiSkill environments
- **Novel RL Algorithms**: Exploration of cutting-edge approaches including NaiviBridger and Visual-RFT
- **Representation Learning**: Integration of R3M pretrained visual encoders with SAC for improved sample efficiency
- **Real-World Deployment**: Jetbot robot experiments with SAC-based navigation policies

---

## Repository Structure

```
Summer-ManiSkill/
├── Baseline Benchmarking/          # Standard RL algorithm implementations
│   ├── ppo/                        # Proximal Policy Optimization
│   ├── sac/                        # Soft Actor-Critic
│   ├── ddpg/                       # Deep Deterministic Policy Gradient
│   ├── tdmpc2/                     # TD-MPC2 (Model-based RL)
│   ├── act/                        # Action Chunking Transformer
│   └── bc/                         # Behavior Cloning
│
├── Novel-rl-algorithms/            # Cutting-edge RL approaches
│   ├── NaiviBridger/               # Diffusion Policy + RL hybrid
│   └── Visual-RFT/                 # Visual Reinforcement Fine-Tuning
│
├── Rl-r3m-integration/             # Pretrained representation + RL
│   └── r3m_sac/                    # R3M encoder with SAC
│
└── navisim_jetbot_proj/            # Real-world robot experiments
    └── navisim_project/            # Jetbot navigation with SAC
```

---

## Module 1: Baseline RL Benchmarking

> **My Contributions**: Independently implemented and tuned 6 RL/IL algorithms on ManiSkill environments, established a complete benchmarking framework, and systematically compared algorithm performance across different tasks.

Systematic evaluation of standard RL algorithms on ManiSkill robotic manipulation tasks (e.g., PickCube, LiftCube).

### Algorithms Implemented

| Algorithm | Type | Key Features | Location |
|-----------|------|--------------|----------|
| **PPO** | On-policy | Stable training, clipped objective | `Baseline Benchmarking/ppo/` |
| **SAC** | Off-policy | Maximum entropy, automatic temperature | `Baseline Benchmarking/sac/` |
| **DDPG** | Off-policy | Deterministic policy, continuous control | `Baseline Benchmarking/ddpg/` |
| **TD-MPC2** | Model-based | World model, planning | `Baseline Benchmarking/tdmpc2/` |
| **ACT** | Imitation | Action chunking, transformer | `Baseline Benchmarking/act/` |
| **BC** | Imitation | Behavioral cloning baseline | `Baseline Benchmarking/bc/` |

### Key Experiments
- Hyperparameter tuning across different tasks
- Comparison of sample efficiency and final performance
- Analysis of training stability and convergence
- RGBD observation support for visual RL

### Results
Training curves and evaluation videos are available in most of the algorithm's subdirectory. Additional experimental results and performance comparisons can be found in this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1yJho8s_5OU8oJKEIL1gGoRp0A-1GxLsRb7snu41LZ8w/edit?gid=0#gid=0).

---

## Module 2: Novel RL Algorithms

> **My Contributions**: Explored and implemented cutting-edge RL algorithms (NaiviBridger, Visual-RFT), integrated them with ManiSkill environments, and analyzed their advantages, limitations, and failure modes compared to baselines.

Exploration of state-of-the-art RL approaches beyond standard baselines.

### NaiviBridger
> Location: `Novel-rl-algorithms/NaiviBridger/`

A hybrid approach combining **Diffusion Policy** with reinforcement learning for manipulation tasks.

**Features:**
- Diffusion-based action generation
- RL fine-tuning pipeline
- Support for PickCube and other ManiSkill tasks
- Dataset collection and conversion utilities

**Key Files:**
- `run_navibridger_maniskill.py` - Main training script
- `diffusion_policy/` - Diffusion model implementation
- `rl.py` - RL training utilities

### Visual-RFT (Visual Reinforcement Fine-Tuning)
> Location: `Novel-rl-algorithms/Visual-RFT/`

Integration of visual language models with RL for improved manipulation.

**Features:**
- GRPO (Group Relative Policy Optimization) training
- Visual representation learning
- ManiSkill environment integration

---

## Module 3: R3M + RL Integration

> **My Contributions**: Designed and implemented the integration pipeline between R3M pretrained visual encoder and SAC, developed embedding wrappers, conducted ablation experiments with frozen/fine-tuned encoders, and validated the sample efficiency improvements from pretrained representations.

> Location: `Rl-r3m-integration/r3m_sac/`

Integration of **R3M** (Reusable Representations for Robotic Manipulation) pretrained visual encoders with SAC for improved sample efficiency in visual RL.

### Approach
- **Pretrained Encoder**: R3M ResNet encoder trained on Ego4D human video data
- **RL Algorithm**: Soft Actor-Critic with frozen/fine-tuned encoder options
- **Task**: PickCube manipulation in ManiSkill

### Experiments
| Configuration | Description |
|---------------|-------------|
| Frozen R3M | Fixed pretrained features, only policy trained |
| Fine-tuned R3M | End-to-end training with encoder |
| Baseline (no R3M) | Standard CNN encoder from scratch |

### Key Findings
- R3M embeddings provide meaningful visual representations
- Improved sample efficiency compared to training from scratch
- Analysis of representation quality for manipulation tasks

### Implementation Details
- Custom wrapper for R3M embedding pipeline
- Integration with ManiSkill observation space
- Training curve logging and visualization

---

## Module 4: Real-World Jetbot Experiments

> **My Contributions**: Built the NaviSim simulation environment, trained SAC navigation policies, deployed policies from simulation to physical Jetbot robot, and addressed sim-to-real gap challenges.

> Location: `navisim_jetbot_proj/navisim_project/`

Deployment of SAC-based navigation policies on a physical **Jetbot** robot.

### Project Overview
Real-world embodied RL experiments demonstrating sim-to-real transfer and practical deployment.

### Features
- **NaviSim Environment**: Custom navigation simulation for Jetbot
- **SAC Training**: Policy learning for navigation tasks
- **Real Robot Deployment**: Transfer to physical Jetbot hardware

### Key Components
- `train_navisim_v2.py` - Main training script
- `configs/` - Training configurations
- `policies/` - Policy network architectures
- `tasks/` - Task definitions

### Challenges Addressed
- Sim-to-real gap mitigation
- Sensor noise handling
- Real-time inference optimization

---

## Technical Skills Demonstrated

### Reinforcement Learning
- On-policy (PPO) and Off-policy (SAC, DDPG) algorithms
- Model-based RL (TD-MPC2)
- Imitation learning (BC, ACT)
- Reward shaping and curriculum learning

### Deep Learning
- PyTorch implementation
- Transformer architectures (ACT)
- Diffusion models (NaiviBridger)
- CNN and ResNet encoders

### Robotics
- ManiSkill simulation environment
- Real robot (Jetbot) deployment
- Visual observation processing (RGB, RGBD)
- Sim-to-real transfer

### Engineering
- Experiment management and logging
- Hyperparameter tuning
- Code organization and documentation
- Version control (Git)

---

## Experimental Results Summary

| Module | Tasks | Key Metrics |
|--------|-------|-------------|
| Baseline Benchmarking | PickCube, LiftCube | Success rate, sample efficiency |
| Novel Algorithms | PickCube | Comparison with baselines |
| R3M Integration | PickCube | Representation quality, learning speed |
| Jetbot | Navigation | Real-world success rate |

*Detailed results, training curves, and videos available in respective module directories.*

---

## Getting Started

### Environment Setup & Dependencies

Please refer to the README in each subdirectory for environment configuration and dependency installation to avoid version conflicts.

| Module | Setup Guide |
|--------|-------------|
| Baseline Benchmarking | `README.md` in each algorithm directory |
| NaiviBridger | `Novel-rl-algorithms/NaiviBridger/README.md` |
| Visual-RFT | `Novel-rl-algorithms/Visual-RFT/README.md` |
| R3M + SAC | `Rl-r3m-integration/r3m_sac/R3M_SAC_PickCube_README.md` |
| Jetbot NaviSim | `navisim_jetbot_proj/navisim_project/NaviSim_README.md` |

---

## References

- [ManiSkill](https://github.com/haosulab/ManiSkill) - Robotic manipulation benchmark
- [R3M](https://github.com/facebookresearch/r3m) - Pretrained visual representations
- [TD-MPC2](https://github.com/nicklashansen/tdmpc2) - Model-based RL
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) - Diffusion for robotics

---

## 👤 Author

**Summer 2024 Research Project**

*This repository showcases research on reinforcement learning for robotic manipulation, demonstrating proficiency in algorithm implementation, experimental design, and real-world deployment.*

**Personal Summary**
*Over the past several months, my work has focused on understanding how reinforcement learning can be used effectively for robotic manipulation. I built and compared baseline algorithms—such as PPO, SAC, DDPG, TD-MPC2, ACT, and several newer approaches—to study how different methods behave across ManiSkill tasks in terms of sample efficiency, stability, and generalization.

A central part of my research examined how pretrained visual representations can support policy learning. By integrating the R3M encoder with SAC, I saw clear improvements in learning speed and robustness, especially in visually complex settings. This experience strengthened my interest in connecting representation learning with decision-making.I also worked on embodied RL by developing a navigation pipeline for Jetbot robots.

Together, these projects gave me a clearer view of how algorithms, visual representations, and physical embodiment interact in robot learning systems. Going forward, I hope to explore more on agentic systems—that can help robots learn efficiently from limited supervision and operate reliably in open-ended environments.

---

## License

This project is for academic and research purposes.
