import json
from tqdm import tqdm


def load_candidates(filepath, limit=None):
    candidates = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            candidates.append(json.loads(line))

            # stop if limit reached
            if limit is not None and len(candidates) >= limit:
                break

    return candidates


if __name__ == "__main__":
    filepath = "../data/raw/candidates.jsonl"
    candidates = load_candidates(filepath, limit=3)

    print("Number of candidates loaded:", len(candidates))
    print("\nTop-level keys:")
    print(candidates[0].keys())

    import pprint
    pp = pprint.PrettyPrinter(indent=2)

    print("\nPROFILE:")
    pp.pprint(candidates[0]["profile"])

    print("\nCAREER HISTORY:")
    pp.pprint(candidates[0]["career_history"][:1])  # first job only

    print("\nSKILLS:")
    pp.pprint(candidates[0]["skills"])

    print("\nREDROB SIGNALS:")
    pp.pprint(candidates[0]["redrob_signals"])