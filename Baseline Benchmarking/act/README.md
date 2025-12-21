# Action Chunking with Transformers (ACT)

This project implements **Action Chunking with Transformers (ACT)** for the ManiSkill task **PickCube-v1**, based on the paper *“Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware”*.  
The implementation is adapted from the original open-source repository: https://github.com/tonyzhaozh/act.

---

## 🚀 Features

- Train ACT on **ManiSkill – PickCube-v1**
- Learn from **offline demonstrations** (motion planning / teleoperation)
- Configurable `--max-episode-steps` to support long demonstration trajectories
- Built-in logging, evaluation, and reproducible experiment configs

---

## ⚙️ Environment Setup (act-ms)

We recommend using Conda to create the environment.  
Save the provided environment configuration as `act-ms.yaml`, then run:

```bash
conda env create -f act-ms.yaml
conda activate act-ms
pip install -e .
```

This assumes the project root contains a `setup.py` or `pyproject.toml`.

---

## 📂 Project Structure (Example)

```bash
act-ms/
│
├── act_ms/                 # Source: models, training logic, utils
│   ├── train.py            # Main training script
│   ├── models/             
│   ├── utils/
│   └── ...
│
├── act-ms.yaml             # Conda environment definition
├── README.md               # Documentation
└── ...
```

---

## 📦 Imitation Learning Setup

Before training, read ManiSkill's imitation learning documentation:

https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html

It explains:

- How to download demonstration datasets  
- How to preprocess them  
- How to evaluate and compare imitation learning methods  
- Tips for improving performance and avoiding common pitfalls  

---

## 🏋️ Training ACT

ACT is trained from demonstration trajectories.

> Some demonstration types (e.g., motion-planning or teleoperation) may exceed the default environment horizon.  
> Since imitation learning reproduces the timing of the demonstrations, set `--max-episode-steps` to **approximately 2× the mean demonstration length**.

Example: Training ACT on **PickCube-v1** with **100 motion-planning state demonstrations**:

```bash
seed=1
demos=100
python train.py --env-id PickCube-v1   --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5   --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num_demos $demos --max_episode_steps 100   --total_iters 30000 --log_freq 100 --eval_freq 5000   --exp-name=act-PickCube-v1-state-${demos}_motionplanning_demos-$seed   --track    # Enable wandb tracking (optional)
```

---

## 🧪 Evaluation & Visualization

Depending on your implementation, evaluation features may include:

- Periodic evaluation of policy success rate and episode returns  
- Rendering PickCube-v1 rollouts to **videos or GIFs**
- Logging curves and metrics with TensorBoard or Weights & Biases  

Additional tools such as `eval.py` or `visualize.py` may be included for rollout visualization.

---

## 📖 Citation

If you use ACT in your research, please cite:

```bibtex
@inproceedings{DBLP:conf/rss/ZhaoKLF23,
  author       = {Tony Z. Zhao and
                  Vikash Kumar and
                  Sergey Levine and
                  Chelsea Finn},
  title        = {Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  booktitle    = {Robotics: Science and Systems XIX},
  year         = {2023},
  doi          = {10.15607/RSS.2023.XIX.016}
}
```
