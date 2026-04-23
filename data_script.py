import json
from tqdm import tqdm

INPUT_PATH  = "data/model_annotations.aligned.paired.jsonl"
OUTPUT_PATH = "data/clean_single_annotation.jsonl"

DIMS = ["coherence", "consistency", "fluency", "relevance"]

def process_sample(sample):
    expert = sample.get("expert_annotations", [])
    turker = sample.get("turker_annotations", [])

    # select ONE annotation
    if expert:
        ann = expert[0]
        source = "expert"
    elif turker:
        ann = turker[0]
        source = "turker"
    else:
        return None

    # extract scores (ensure all dims exist)
    clean_scores = {}
    for d in DIMS:
        clean_scores[d] = float(ann.get(d, 3.0))

    # build cleaned sample
    new_sample = {
        "text": sample["text"],
        "summary": sample["decoded"],
        "expert_annotations": clean_scores,
        "source": source   # optional (useful for debugging)
    }

    return new_sample


print("\n[Processing dataset...]")

count_in = 0
count_out = 0
expert_count = 0
turker_count = 0

with open(INPUT_PATH, "r", encoding="utf-8", errors="ignore") as fin, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for line in tqdm(fin):
        count_in += 1

        try:
            sample = json.loads(line.strip())
        except:
            continue

        new_sample = process_sample(sample)
        if new_sample is None:
            continue

        if new_sample["source"] == "expert":
            expert_count += 1
        else:
            turker_count += 1

        fout.write(json.dumps(new_sample) + "\n")
        count_out += 1


print("\nDONE")
