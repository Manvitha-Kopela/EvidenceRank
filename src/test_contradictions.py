from parser import load_candidates
from features import contradiction_features

candidate = load_candidates("../data/raw/candidates.jsonl", limit=1)[0]

print(contradiction_features(candidate))