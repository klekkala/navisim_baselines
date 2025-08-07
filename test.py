from huggingface_hub import hf_hub_download
import os, shutil

task = 'PickCube-v1'
repo = 'haosulab/ManiSkill_Demonstrations'
filename = f'demos/{task}/motionplanning/trajectory.h5'

# 下载 .h5 文件
path = hf_hub_download(
    repo_id=repo,
    filename=filename,
    repo_type='dataset',
    use_auth_token=False
)

# 移动到正确位置
target = os.path.expanduser(f'~/.maniskill/demos/{task}/motionplanning/')
os.makedirs(target, exist_ok=True)
shutil.copy(path, os.path.join(target, 'trajectory.h5'))

print(f"✅ {task} 的 trajectory.h5 下载完成")
