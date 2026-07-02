from jd_parser import parse_job_description
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from weight_generator import generate_behavior_weights
import numpy as np
import os

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDINGS_CACHE = os.path.join(BASE_DIR, "outputs", "candidate_embeddings.npy")
EMBEDDINGS_IDS_CACHE = os.path.join(BASE_DIR, "outputs", "candidate_ids.npy")

_embedding_map = {}
title_embedding_cache = {}

JD_INFO = None
JD_SKILL_MAP = {}
JD_TITLE_EMBEDDING = None
jd_embedding = None
BEHAVIOR_WEIGHTS = None


SKILL_ALIASES = {
    "postgresql": ["postgresql", "postgres", "psql"],
    "machine learning": ["machine learning", "ml"],
    "kubernetes": ["kubernetes", "k8s"],
    "amazon web services": ["aws", "amazon web services"],
    "javascript": ["javascript", "js"],
    "artificial intelligence": ["ai", "artificial intelligence"]
}


def ensure_initialized():
    if JD_INFO is None:
        raise RuntimeError(
            "Features not initialized. Call initialize_features(jd_path) first."
        )


def get_jd_info():
    ensure_initialized()
    return JD_INFO


def initialize_features(jd_path):
    global JD_INFO, JD_SKILL_MAP, JD_TITLE_EMBEDDING, jd_embedding
    global title_embedding_cache, BEHAVIOR_WEIGHTS

    JD_INFO = parse_job_description(jd_path)

    print("Using JD:", jd_path)
    print("Parsed JD:", JD_INFO)

    JD_SKILL_MAP = {}

    for skill in JD_INFO["required_skills"]:
        skill = skill.lower().strip()
        if skill in SKILL_ALIASES:
            JD_SKILL_MAP[skill] = SKILL_ALIASES[skill]
        else:
            JD_SKILL_MAP[skill] = [skill]

    job_title = JD_INFO.get("job_title", "Unknown Role")
    seniority = JD_INFO.get("seniority", "")
    skills = JD_INFO.get("required_skills", [])

    JD_TITLE_EMBEDDING = model.encode(job_title)

    jd_text = f"""
    Job Title: {job_title}
    Seniority: {seniority}
    Required skills: {', '.join(skills)}
    Company style: {JD_INFO.get('company_style', '')}
    """

    jd_embedding = model.encode(jd_text)

    # Reset per-JD caches so stale data from a previous JD never leaks in
    title_embedding_cache = {}
    BEHAVIOR_WEIGHTS = generate_behavior_weights(JD_INFO)


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

    texts = [build_candidate_text(c) for c in candidates]
    ids = [c["candidate_id"] for c in candidates]

    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    np.save(EMBEDDINGS_CACHE, embeddings)
    np.save(EMBEDDINGS_IDS_CACHE, np.array(ids))

    _embedding_map = dict(zip(ids, embeddings))


def compute_semantic_score(candidate):
    ensure_initialized()

    cid = candidate["candidate_id"]

    if cid in _embedding_map:
        candidate_embedding = _embedding_map[cid]
    else:
        candidate_embedding = model.encode(build_candidate_text(candidate))

    score = cosine_similarity(
        [jd_embedding],
        [candidate_embedding]
    )[0][0]

    return float(score)


def compute_title_similarity(candidate):
    ensure_initialized()

    candidate_title = candidate["profile"].get("current_title", "")

    if candidate_title in title_embedding_cache:
        candidate_emb = title_embedding_cache[candidate_title]
    else:
        candidate_emb = model.encode(candidate_title)
        title_embedding_cache[candidate_title] = candidate_emb

    score = cosine_similarity(
        [JD_TITLE_EMBEDDING],
        [candidate_emb]
    )[0][0]

    return float(score)


def compute_skill_overlap(skill_names):
    ensure_initialized()

    matched = 0
    candidate_skills = [s.lower() for s in skill_names]

    for _, aliases in JD_SKILL_MAP.items():
        found = False

        for alias in aliases:
            if any(alias in skill for skill in candidate_skills):
                found = True
                break

        if found:
            matched += 1

    if len(JD_SKILL_MAP) == 0:
        return 0

    return matched / len(JD_SKILL_MAP)


def normalize(value, min_val, max_val):
    if value is None:
        return 0
    return max(0, min(1, (value - min_val) / (max_val - min_val)))


def qualification_features(candidate):
    ensure_initialized()

    profile = candidate["profile"]
    years_exp = profile.get("years_of_experience", 0)
    skill_names = [s["name"].lower() for s in candidate.get("skills", [])]

    return {
        "years_experience": years_exp,
        "skill_overlap": compute_skill_overlap(skill_names),
        "semantic_score": compute_semantic_score(candidate),
        "title_score": compute_title_similarity(candidate)
    }


def behavioral_features(candidate):
    ensure_initialized()

    signals = candidate["redrob_signals"]
    weights = BEHAVIOR_WEIGHTS

    github = normalize(signals.get("github_activity_score", 0), 0, 100)
    response = signals.get("recruiter_response_rate", 0)
    completion = signals.get("interview_completion_rate", 0)
    saves = normalize(signals.get("saved_by_recruiters_30d", 0), 0, 50)

    behavior_score = (
        weights["github"] * github +
        weights["response"] * response +
        weights["completion"] * completion +
        weights["saves"] * saves
    )

    return {
        "behavior_score": behavior_score,
        "github_score": github,
        "response_rate": response,
        "completion_rate": completion,
        "recruiter_saves": saves
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
        assessment_quality = (
            sum(assessment_scores.values()) /
            (len(assessment_scores) * 100)
        )

    evidence_density = (
        0.25 * github +
        0.25 * profile_complete +
        0.25 * response +
        0.15 * completion +
        0.10 * assessment_quality
    )

    return {"evidence_density": evidence_density}


def detect_honeypot(candidate):
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    flags = 0

    zero_duration_experts = sum(
        1 for s in skills
        if s.get("proficiency") == "advanced"
        and s.get("duration_months", 1) == 0
    )

    if zero_duration_experts >= 3:
        flags += 1

    years_exp = profile.get("years_of_experience", 0)

    for job in profile.get("career_history", []):
        start = job.get("start_date", "")
        try:
            start_year = int(start[:4])
            company_age = 2026 - start_year

            if years_exp > 0 and company_age < years_exp * 0.4:
                flags += 1
                break
        except Exception:
            pass

    return flags > 0