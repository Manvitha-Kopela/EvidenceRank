import pandas as pd
import json
from parser import load_candidates
from features import (
    qualification_features, behavioral_features,
    contradiction_features, confidence_features,
    load_embedding_cache, detect_honeypot
)
from scorer import calculate_scores, calculate_risk

def generate_reasoning(candidate, q, b, c):
    profile = candidate["profile"]
    title = profile.get("current_title", "")
    years = profile.get("years_of_experience", 0)
    skills = candidate.get("skills", [])
    signals = candidate["redrob_signals"]
    
    # Get advanced skills
    advanced_skills = [s["name"] for s in skills if s.get("proficiency") == "advanced"][:3]
    all_skills = [s["name"].lower() for s in skills]
    
    response = signals.get("recruiter_response_rate", 0)
    github = signals.get("github_activity_score", 0)
    saves = signals.get("saved_by_recruiters_30d", 0)
    
    # JD relevance signals
    jd_keywords = ["rag", "retrieval", "vector", "embedding", "llm", "ranking", 
                   "faiss", "pinecone", "milvus", "weaviate", "nlp", "search"]
    matched_jd = [s for s in advanced_skills if any(k in s.lower() for k in jd_keywords)]
    
    # Build first sentence — role fit
    if matched_jd:
        first = f"{title} with {years:.1f} yrs and direct JD-relevant expertise in {', '.join(matched_jd)}."
    elif advanced_skills:
        first = f"{title} with {years:.1f} yrs; advanced skills in {', '.join(advanced_skills)} with partial JD overlap."
    else:
        first = f"{title} with {years:.1f} yrs; limited direct skill match to JD requirements."
    
    # Build second sentence — behavioral signals + honest concerns
    parts = []
    
    if response >= 0.7:
        parts.append(f"high recruiter engagement ({response:.2f} response rate)")
    elif response >= 0.4:
        parts.append(f"moderate recruiter engagement ({response:.2f} response rate)")
    else:
        parts.append(f"low recruiter response rate ({response:.2f}) is a concern")
    
    if github >= 60:
        parts.append(f"strong GitHub activity ({github:.0f})")
    elif github < 20:
        parts.append(f"weak GitHub signal ({github:.0f})")
    
    if saves >= 10:
        parts.append(f"saved by {saves:.0f} recruiters recently")
    
    if years < 5 and "staff" in title.lower():
        parts.append("experience level below typical staff threshold")
    elif years >= 8:
        parts.append(f"senior experience level supports seniority match")
    
    if c["contradiction_score"] > 0:
        parts.append(f"flag: {c['reasons'][0].lower()}")
    
    second = "; ".join(parts).capitalize() + "."
    
    return f"{first} {second}"

print("Loading candidates...")
candidates = load_candidates("../data/raw/candidates.jsonl")

print("Loading embedding cache...")
load_embedding_cache()

results = []
honeypot_count = 0

print("Scoring candidates...")
for candidate in candidates:
    try:
        if detect_honeypot(candidate):
            honeypot_count += 1
            continue

        q = qualification_features(candidate)
        b = behavioral_features(candidate)
        c = contradiction_features(candidate)
        conf = confidence_features(candidate)
        scores = calculate_scores(q, b, c, conf)

        results.append({
            "candidate_id": candidate["candidate_id"],
            "fit_score": scores["fit_score"],
            "reasoning_data": (candidate, q, b, c)
        })
    except Exception as e:
        print(f"Error {candidate['candidate_id']}: {e}")

print(f"Honeypots filtered: {honeypot_count}")

# Sort and take top 100
results.sort(key=lambda x: (-x["fit_score"], x["candidate_id"]))
top100 = results[:100]

rows = []
for rank, item in enumerate(top100, 1):
    candidate, q, b, c = item["reasoning_data"]
    rows.append({
        "candidate_id": candidate["candidate_id"],
        "rank": rank,
        "score": round(item["fit_score"], 6),  # more decimal places
        "reasoning": generate_reasoning(candidate, q, b, c)
    })

df = pd.DataFrame(rows)
df.to_csv("../outputs/final_submission.csv", index=False)

print("\nTop 10:")
print(df.head(10))
print(f"\nTotal rows: {len(df)}")
print("Saved to ../outputs/final_submission.csv")