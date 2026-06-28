from parser import load_candidates
from features import qualification_features

candidates = load_candidates("../data/raw/candidates.jsonl", limit=1)

candidate = candidates[0]

print(qualification_features(candidate))