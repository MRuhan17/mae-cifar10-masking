import os
import json

RESULTS_DIR = "results/linear_probe"

def extract_accuracy(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()
    for line in reversed(lines):
        if "Final linear probe accuracy" in line:
            return float(line.strip().split()[-1].replace("%", ""))
    return None

rows = []

for folder in os.listdir(RESULTS_DIR):
    sub = os.path.join(RESULTS_DIR, folder)
    if not os.path.isdir(sub):
        continue

    mask_type, mask_ratio = folder.split("_")
    log_path = os.path.join(sub, "log.txt")
    acc = extract_accuracy(log_path)

    rows.append([mask_type, mask_ratio, acc])

rows.sort()

# print table
print("mask_type,mask_ratio,accuracy")
for r in rows:
    print(f"{r[0]},{r[1]},{r[2]}")
