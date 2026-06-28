# save as check_honeypots.py in src/
import json
import pandas as pd

top100 = pd.read_csv("../outputs/final_submission.csv")
top_ids = set(top100["candidate_id"])

with open("../data/raw/candidates.jsonl", "r") as f:
    for line in f:
        c = json.loads(line)
        if c["candidate_id"] not in top_ids:
            continue
        
        skills = c.get("skills", [])
        years = c["profile"].get("years_of_experience", 0)
        
        zero_experts = [s["name"] for s in skills 
                       if s.get("proficiency") == "advanced" 
                       and s.get("duration_months", 1) == 0]
        
        if zero_experts:
            print(f"{c['candidate_id']} | {years}yrs | zero-duration advanced: {zero_experts}")