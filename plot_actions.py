import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Find the newest actions_and_joint_positions.npz file in all play_data folders
search_root = "logs/rsl_rl"
pattern = os.path.join(search_root, "*", "*", "play_data", "actions.npz")
files = glob.glob(pattern)
if not files:
    raise FileNotFoundError("No actions_and_joint_positions.npz file found in any play_data directory.")
# Pick the latest modified file
latest_file = max(files, key=os.path.getmtime)
print(f"Using data file: {latest_file}")

data = np.load(latest_file)
actions = data["actions"]

plt.figure(figsize=(12, 6))
for i in range(actions.shape[1]):
    plt.plot(actions[:, i], label=f"Action {i}")

plt.xlabel("Timestep")
plt.ylabel("Action Value")
plt.title("Actions over Time")
plt.legend()
plt.tight_layout()
plt.show()