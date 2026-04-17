import subprocess, os, random, sys

API_KEY = sys.argv[1]
DATASET = "VLM-Physics-Finetuning-Data/material_dataset"
TARGET = 7
found = []

candidates = list(range(0, 5000, 50))
random.seed(42)
random.shuffle(candidates)

for idx in candidates:
    if len(found) >= TARGET:
        break
    img = f"{DATASET}/mat_img_{idx}.jpg"
    if not os.path.exists(img):
        continue
    out = f"report_{idx}.json"
    print(f"\n--- Testing mat_img_{idx}.jpg ---")
    r = subprocess.run(["python", "agent.py", "--image", img, "--mp-api-key", API_KEY, "--output", out],
                       capture_output=True, text=True, timeout=120)
    if "Extracted materials: []" in r.stdout or "Extracted materials:" not in r.stdout:
        print("  No materials, skip.")
        if os.path.exists(out): os.remove(out)
    else:
        for line in r.stdout.split("\n"):
            if "Extracted materials:" in line:
                print(f"  {line.strip()}")
        found.append(idx)
        print(f"  FOUND! ({len(found)}/{TARGET})")

print(f"\nDone. Found: {[f'mat_img_{i}.jpg' for i in found]}")
