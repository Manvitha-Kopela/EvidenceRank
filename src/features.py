from datetime import datetime
from jd_parser import parse_job_description
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
JD_INFO = parse_job_description("../data/raw/job_description.docx")

JD_SKILL_MAP = {
    "python": ["python"],
    "retrieval": ["retrieval", "bm25", "rag"],
    "ranking": ["ranking", "reranking"],
    "vector_db": ["milvus", "pinecone", "faiss", "vector database"],
    "ml": ["machine learning", "ml", "nlp", "llm", "fine-tuning"]
}

JD_TEXT = """
Staff Machine Learning Engineer with expertise in RAG, retrieval systems,
vector databases, embeddings, ranking and LLM systems.
Building production-scale ML pipelines for search and recommendation.
"""

jd_embedding = model.encode(JD_TEXT)

EMBEDDINGS_CACHE = "../outputs/candidate_embeddings.npy"
EMBEDDINGS_IDS_CACHE = "../outputs/candidate_ids.npy"

_embedding_map = {}

def load_embedding_cache():
    global _embedding_map
    if os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(EMBEDDINGS_IDS_CACHE):
        embeddings = np.load(EMBEDDINGS_CACHE)
        ids = np.load(EMBEDDINGS_IDS_CACHE, allow_pickle=True)
        _embedding_map = dict(zip(ids, embeddings))
        print(f"Loaded {len(_embedding_map)} cached embeddings.")

def build_candidate_text(candidate):
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "")
    summary = profile.get("summary", "")
    history = profile.get("career_history", [])
    recent_desc = history[0].get("description", "") if history else ""
    skills = [s["name"] for s in candidate.get("skills", [])]
    return f"{title}. {summary} {recent_desc} Skills: {', '.join(skills[:15])}"

def precompute_embeddings(candidates):
    global _embedding_map
    print(f"Precomputing embeddings for {len(candidates)} candidates...")
    texts = [build_candidate_text(c) for c in candidates]
    ids = [c["candidate_id"] for c in candidates]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    np.save(EMBEDDINGS_CACHE, embeddings)
    np.save(EMBEDDINGS_IDS_CACHE, np.array(ids))
    _embedding_map = dict(zip(ids, embeddings))
    print("Embeddings cached.")

def compute_semantic_score(candidate):
    cid = candidate["candidate_id"]
    if cid in _embedding_map:
        candidate_embedding = _embedding_map[cid]
    else:
        text = build_candidate_text(candidate)
        candidate_embedding = model.encode(text)
    score = cosine_similarity([jd_embedding], [candidate_embedding])[0][0]
    return float(score)

def compute_skill_overlap(skill_names):
    matched = 0
    for category, aliases in JD_SKILL_MAP.items():
        found = any(alias in skill for skill in skill_names for alias in aliases)
        if found:
            matched += 1
    return matched / len(JD_SKILL_MAP)

def normalize(value, min_val, max_val):
    if value is None:
        return 0
    return max(0, min(1, (value - min_val) / (max_val - min_val)))

def qualification_features(candidate):
    profile = candidate["profile"]
    years_exp = profile.get("years_of_experience", 0)
    skill_names = [s["name"].lower() for s in candidate.get("skills", [])]
    skill_overlap = compute_skill_overlap(skill_names)
    semantic_score = compute_semantic_score(candidate)
    title = profile.get("current_title", "").lower()
    jd_seniority = JD_INFO["seniority"]

    title_score = 0
    if "engineer" in title:
        title_score += 0.5
    elif "developer" in title:
        title_score += 0.4

    if jd_seniority == "staff":
        if "staff" in title or "principal" in title:
            title_score += 0.5
    elif jd_seniority == "senior":
        if "senior" in title:
            title_score += 0.5
    else:
        title_score += 0.3

    title_score = min(title_score, 1)

    return {
        "years_experience": years_exp,
        "skill_overlap": skill_overlap,
        "semantic_score": semantic_score,
        "title_score": title_score
    }

def behavioral_features(candidate):
    signals = candidate["redrob_signals"]
    github = signals.get("github_activity_score", 0)
    response = signals.get("recruiter_response_rate", 0)
    completion = signals.get("interview_completion_rate", 0)
    saves = signals.get("saved_by_recruiters_30d", 0)
    return {
        "github_score": normalize(github, 0, 100),
        "response_rate": response,
        "completion_rate": completion,
        "recruiter_saves": normalize(saves, 0, 50)
    }

def contradiction_features(candidate):
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    skills = candidate.get("skills", [])

    contradictions = 0
    reasons = []

    title = profile.get("current_title", "").lower()
    years_exp = profile.get("years_of_experience", 0)
    github = signals.get("github_activity_score", 0)
    response = signals.get("recruiter_response_rate", 0)
    open_to_work = signals.get("open_to_work_flag", False)
    assessment_scores = signals.get("skill_assessment_scores", {})
    saves = signals.get("saved_by_recruiters_30d", 0)

    if ("ai" in title or "ml" in title) and github < 20:
        contradictions += 1
        reasons.append("AI/ML title but weak GitHub activity")

    if open_to_work and response < 0.4:
        contradictions += 1
        reasons.append("Open to work but low recruiter response")

    if "staff" in title and years_exp < 5:
        contradictions += 1
        reasons.append("Staff title but low experience")

    for skill in skills:
        endorsements = skill.get("endorsements", 0)
        skill_name = skill.get("name")
        if skill_name in assessment_scores:
            score = assessment_scores[skill_name]
            if endorsements > 20 and score < 50:
                contradictions += 1
                reasons.append("High endorsements but weak skill assessment")
                break

    if saves > 10 and response < 0.5:
        contradictions += 1
        reasons.append("High recruiter demand but low engagement")

    return {
        "contradiction_score": min(contradictions, 5),
        "reasons": reasons
    }

def confidence_features(candidate):
    signals = candidate["redrob_signals"]
    github = normalize(signals.get("github_activity_score", 0), 0, 100)
    profile_complete = normalize(signals.get("profile_completeness_score", 0), 0, 100)
    response = signals.get("recruiter_response_rate", 0)
    completion = signals.get("interview_completion_rate", 0)
    assessment_scores = signals.get("skill_assessment_scores", {})

    assessment_quality = 0
    if assessment_scores:
        assessment_quality = sum(assessment_scores.values()) / (len(assessment_scores) * 100)

    evidence_density = (
        0.25 * github
        + 0.25 * profile_complete
        + 0.25 * response
        + 0.15 * completion
        + 0.10 * assessment_quality
    )
    return {"evidence_density": evidence_density}
def detect_honeypot(candidate):
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    flags = 0

    # Check: expert proficiency with 0 duration months
    zero_duration_experts = sum(
        1 for s in skills
        if s.get("proficiency") == "advanced" and s.get("duration_months", 1) == 0
    )
    if zero_duration_experts >= 3:
        flags += 1

    # Check: experience years > tenure at any single company
    years_exp = profile.get("years_of_experience", 0)
    for job in profile.get("career_history", []):
        start = job.get("start_date", "")
        try:
            start_year = int(start[:4])
            company_age = 2026 - start_year
            if years_exp > 0 and company_age < years_exp * 0.4:
                flags += 1
                break
        except:
            pass

    return flags > 0