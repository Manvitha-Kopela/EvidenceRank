from parser import load_candidates
from collections import Counter

candidates = load_candidates("../data/raw/candidates.jsonl")

counter = Counter()

for c in candidates:
    title = c["profile"].get("current_title", "").lower()

    if "marketing" in title:
        counter["marketing"] += 1
    elif "sales" in title:
        counter["sales"] += 1
    elif "engineer" in title:
        counter["engineer"] += 1
    elif "designer" in title:
        counter["designer"] += 1
    else:
        counter["other"] += 1

print(counter)