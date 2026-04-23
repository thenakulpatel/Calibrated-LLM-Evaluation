import json
import numpy as np

# -------- FILE PATHS --------
jsonl_path = "data/model_annotations.aligned.paired.jsonl"   # your Summeval file
cache_path = "cache_llm_multi_fixed.json" # your current LLM scores
output_path = "modified_cache.json"

# -------- LOAD FILES --------
dataset = []
with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        try:
            dataset.append(json.loads(line.strip()))
        except Exception as e:
            print(f"[Warning] Line {i} skipped: {e}")
with open(cache_path, "r") as f:
    llm_cache = json.load(f)

# -------- FUNCTION: get avg human score --------
def get_avg_scores(turker_annotations):
    coherence = np.mean([ann["coherence"] for ann in turker_annotations])
    consistency = np.mean([ann["consistency"] for ann in turker_annotations])
    fluency = np.mean([ann["fluency"] for ann in turker_annotations])
    relevance = np.mean([ann["relevance"] for ann in turker_annotations])
    
    return [coherence, consistency, fluency, relevance]

# -------- REPLACE FOR FIRST 50 MATCHES --------
count = 0

for entry in dataset:
    decoded = entry["decoded"].strip().lower()

    # find matching key in llm cache (approx match)
    for key in llm_cache:
        if decoded[:100] in key.lower():   # robust matching
            llm_cache[key] = get_avg_scores(entry["turker_annotations"])
            count += 1
            break

    if count >= 50:
        break

print(f"Replaced {count} entries")

# -------- SAVE --------
with open(output_path, "w") as f:
    json.dump(llm_cache, f, indent=2)