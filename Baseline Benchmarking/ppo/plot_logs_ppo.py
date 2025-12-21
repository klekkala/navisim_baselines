import re
import matplotlib.pyplot as plt

log_file_path = "log_ppo.txt"  

reward_list = []
kl_list = []
entropy_loss_list = []
policy_grad_loss_list = []
value_loss_list = []
total_loss_list = []

float_pattern = r"-?\d+(?:\.\d+)?"

with open(log_file_path, "r") as f:
    lines = f.readlines()

for line in lines:
    if "|    ep_rew_mean" in line:
        match = re.search(float_pattern, line)
        if match:
            reward_list.append(float(match.group()))
    elif "|    approx_kl" in line:
        match = re.search(float_pattern, line)
        if match:
            kl_list.append(float(match.group()))
    elif "|    entropy_loss" in line:
        match = re.search(float_pattern, line)
        if match:
            entropy_loss_list.append(float(match.group()))
    elif "|    policy_gradient_loss" in line:
        match = re.search(float_pattern, line)
        if match:
            policy_grad_loss_list.append(float(match.group()))
    elif "|    value_loss" in line:
        match = re.search(float_pattern, line)
        if match:
            value_loss_list.append(float(match.group()))
    elif "|    loss" in line:
        match = re.search(float_pattern, line)
        if match:
            total_loss_list.append(float(match.group()))


max_len = len(reward_list)
def pad_list(lst):
    while len(lst) < max_len:
        lst.append(None)
    return lst

kl_list = pad_list(kl_list)
entropy_loss_list = pad_list(entropy_loss_list)
policy_grad_loss_list = pad_list(policy_grad_loss_list)
value_loss_list = pad_list(value_loss_list)
total_loss_list = pad_list(total_loss_list)

# 绘图
plt.figure(figsize=(16, 10))

plt.subplot(3, 2, 1)
plt.plot(reward_list, label="Episode Reward Mean")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(kl_list, label="Approx KL")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(entropy_loss_list, label="Entropy Loss")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(policy_grad_loss_list, label="Policy Gradient Loss")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(value_loss_list, label="Value Loss")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(total_loss_list, label="Total Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("ppo_training_metrics.png")
print("✅ 图像已保存为 ppo_training_metrics.png")
