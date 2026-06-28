from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

JD_TEXT = """
Staff Machine Learning Engineer focused on retrieval, ranking, and RAG systems.
Required: Python, embeddings, vector databases, semantic search, LLMs,
retrieval augmented generation.
Building production-scale ML pipelines for search and recommendation.
"""

def build_candidate_text(candidate):
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "")
    summary = profile.get("summary", "")
    history = profile.get("career_history", [])
    recent_desc = history[0].get("description", "") if history else ""
    skills = [s["name"] for s in candidate.get("skills", [])]  # fixed: top level
    return f"{title}. {summary} {recent_desc} Skills: {', '.join(skills[:15])}"

def test_on_sample(jsonl_path, n=100):
    candidates = []
    print("Loading candidates...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            candidates.append(json.loads(line))

    print("Encoding JD...")
    jd_embedding = model.encode([JD_TEXT])

    texts = [build_candidate_text(c) for c in candidates]
    print("Encoding candidates...")
    candidate_embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    scores = cosine_similarity(jd_embedding, candidate_embeddings)[0]

    results = sorted(
        zip([c["candidate_id"] for c in candidates], scores),
        key=lambda x: x[1], reverse=True
    )

    print("\nTop 10 semantic matches:")
    for cid, score in results[:10]:
        print(f"  {cid}: {score:.4f}")

    print("\nBottom 5:")
    for cid, score in results[-5:]:
        print(f"  {cid}: {score:.4f}")

    print("\n--- Top candidate profile ---")
    top_id = results[0][0]
    for c in candidates:
        if c["candidate_id"] == top_id:
            p = c["profile"]
            print("Title:", p.get("current_title", "N/A"))
            print("Summary:", p.get("summary", "")[:300])
            print("Skills:", [s["name"] for s in c.get("skills", [])[:8]])
            break

if __name__ == "__main__":
    test_on_sample("../data/raw/candidates.jsonl", n=100)