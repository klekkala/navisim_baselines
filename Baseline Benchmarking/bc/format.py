import h5py
import os
path = os.path.expanduser("~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.state.pd_joint_pos.cpu.h5")
with h5py.File(path, "r") as f:
    traj = f["traj_0"]
    print("Keys in traj_0:", list(traj.keys()))
    print("Keys in traj_0/obs:", list(traj["obs"].keys()))
    for k in traj["obs"].keys():
        print(f"{k} shape: {traj['obs'][k].shape}")
