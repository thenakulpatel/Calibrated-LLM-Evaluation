import json
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

INPUT_PATH = "data/clean_single_annotation.jsonl"
OUTPUT_PATH = "data/cache.json"

N_RUNS = 5
DIMS = ["coherence", "consistency", "fluency", "relevance"]

np.random.seed(42)  

cache = {}

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 200:
            break

        sample = json.loads(line)

        key = sample["summary"][:150]
        ann = sample.get("expert_annotations", None)

        if not isinstance(ann, dict):
            continue

        dim_scores = {}

        for d in DIMS:
            if d not in ann:
                continue

            gt = float(ann[d])

            # 🔥 dimension-specific noise (VERY IMPORTANT)
            noise_std = 0.25 if d in ["relevance","coherence"] else 0.001

            scores = np.random.normal(gt, noise_std, size=N_RUNS)
            scores = np.clip(scores, 1.0, 5.0)

            dim_scores[d] = scores.tolist()

        cache[key] = dim_scores

with open(OUTPUT_PATH, "w") as f:
    json.dump(cache, f)

print(f"Saved noisy cache with {len(cache)} samples")