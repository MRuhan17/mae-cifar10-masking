import os
import json
import matplotlib.pyplot as plt

PRETRAIN_DIR = "results/pretrain"

configs_to_plot = [
    "random_0.75",
    "random_0.9",
    "block_0.6",
    "block_0.9"
]

for cfg in configs_to_plot:
    log_path = os.path.join(PRETRAIN_DIR, cfg, "loss_log.json")

    with open(log_path, "r") as f:
        logs = json.load(f)

    losses = logs["loss"]
    plt.plot(losses, label=cfg)

plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.title("MAE Pretraining Loss Curves")
plt.legend()
plt.tight_layout()

plt.savefig("results/loss_curves.png")
