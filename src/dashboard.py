import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jd_parser import parse_job_description
from features import contradiction_features, confidence_features, qualification_features, behavioral_features, load_embedding_cache
from parser import load_candidates
from scorer import calculate_scores, calculate_risk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "candidates.jsonl")
JD_PATH = os.path.join(BASE_DIR, "data", "raw", "job_backend.docx")
RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "final_submission.csv")

st.set_page_config(page_title="EvidenceRank", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
* { box-sizing: border-box; }
.stApp { background: #080c14; font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] { background: #0d1220 !important; border-right: 1px solid #1e2d4a !important; }
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
.page-header { padding: 32px 0 24px 0; border-bottom: 1px solid #1e2d4a; margin-bottom: 32px; }
.page-title { font-size: 32px; font-weight: 700; color: #f1f5f9; letter-spacing: -1px; line-height: 1.1; }
.page-title span { color: #3b82f6; }
.page-subtitle { font-size: 14px; color: #475569; margin-top: 6px; }
.section-header { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: #3b82f6; margin: 40px 0 16px 0; padding-bottom: 8px; border-bottom: 1px solid #1e2d4a; }
.metric-card { background: #0d1220; border: 1px solid #1e2d4a; border-radius: 8px; padding: 20px; text-align: center; }
.metric-value { font-size: 28px; font-weight: 700; color: #f1f5f9; font-family: 'JetBrains Mono', monospace; line-height: 1; }
.metric-label { font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px; }
.metric-accent { color: #3b82f6; }
.reasoning-box { background: #0d1220; border: 1px solid #1e2d4a; border-left: 3px solid #3b82f6; border-radius: 0 8px 8px 0; padding: 16px 20px; font-size: 14px; color: #94a3b8; line-height: 1.7; }
.jd-pill { display: inline-block; background: #1e2d4a; color: #7dd3fc !important; border: 1px solid #1e40af; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-family: 'JetBrains Mono', monospace; margin: 2px; }
.divider { border: none; border-top: 1px solid #1e2d4a; margin: 32px 0; }
div[data-baseweb="select"] > div { background-color: #0d1220 !important; border-color: #1e2d4a !important; color: #f1f5f9 !important; }
div[data-baseweb="select"] span { color: #f1f5f9 !important; }
div[data-baseweb="popover"] { background-color: #0d1220 !important; }
div[data-baseweb="popover"] * { background-color: #0d1220 !important; color: #94a3b8 !important; }
li[role="option"]:hover { background-color: #1e2d4a !important; color: #f1f5f9 !important; }
.stDataFrame { background: transparent !important; }
[data-testid="stDataFrame"] > div { background: #0d1220 !important; border: 1px solid #1e2d4a !important; border-radius: 8px !important; overflow: hidden; }
iframe { background: #0d1220 !important; }
.hire-badge-green { display:inline-block; background:#052e16; color:#4ade80; border:1px solid #166534; border-radius:20px; padding:4px 12px; font-size:12px; font-weight:600; }
.hire-badge-yellow { display:inline-block; background:#1c1a10; color:#fbbf24; border:1px solid #92400e; border-radius:20px; padding:4px 12px; font-size:12px; font-weight:600; }
.hire-badge-red { display:inline-block; background:#1c0a0a; color:#f87171; border:1px solid #7f1d1d; border-radius:20px; padding:4px 12px; font-size:12px; font-weight:600; }
.gem-card { background:#0d1220; border:1px solid #1e40af; border-left:3px solid #3b82f6; border-radius:8px; padding:14px 18px; margin-bottom:10px; }
.gem-title { font-size:13px; font-weight:600; color:#f1f5f9; margin-bottom:4px; }
.gem-sub { font-size:12px; color:#64748b; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)


def hire_label(fit, risk, contradictions):
    if contradictions >= 2:
        return "High Risk"

    if fit >= 0.74 and risk <= 0.72:
        return "Strong Hire"

    if risk > 0.85:
        return "High Risk"

    return "Needs Review"
def tag(style, icon, text):
    return f'<div style="display:inline-block;{style}border-radius:4px;padding:5px 10px;font-size:12px;margin:3px 3px 3px 0;">{icon} {text}</div>'

TAG_GREEN  = "background:#052e16;color:#4ade80;border:1px solid #166534;"
TAG_ORANGE = "background:#1c1917;color:#fb923c;border:1px solid #7c2d12;"
TAG_RED    = "background:#1c0a0a;color:#f87171;border:1px solid #7f1d1d;"

JD_SKILLS = {
    "python":       ["python"],
    "rag":          ["rag", "retrieval augmented", "retrieval-augmented"],
    "vector db":    ["milvus", "pinecone", "faiss", "weaviate", "qdrant", "vector db", "vector database"],
    "llm":          ["llm", "large language", "gpt", "claude", "gemini"],
    "embeddings":   ["embedding", "sentence-transformer", "minilm", "ada"],
    "nlp":          ["nlp", "natural language", "spacy", "huggingface", "transformers"],
    "ranking":      ["ranking", "reranking", "bm25", "colbert"],
    "retrieval":    ["retrieval", "dense retrieval", "sparse retrieval", "semantic search"],
}


@st.cache_data
def load_all_data():
    load_embedding_cache()
    candidates = load_candidates(DATA_PATH)
    return {c["candidate_id"]: c for c in candidates}

@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)

@st.cache_data
def load_jd():
    return parse_job_description(JD_PATH)

jd = load_jd()
df = load_results()
candidate_map = load_all_data()
candidate_ids = df["candidate_id"].tolist()


def skill_match_for(candidate):
    skills_text = " ".join([s["name"].lower() for s in candidate.get("skills", [])])
    summary = candidate.get("profile", {}).get("summary", "").lower()
    full_text = skills_text + " " + summary
    result = {}
    for jd_skill, aliases in JD_SKILLS.items():
        result[jd_skill] = any(a in full_text for a in aliases)
    return result


def find_hidden_gems(candidate_map, df, n=5):
    """Candidates with medium keyword overlap but high semantic + behavioral scores."""
    gems = []
    for cid, cand in list(candidate_map.items()):
        row = df[df["candidate_id"] == cid]
        if row.empty:
            continue
        rank_val = int(row.iloc[0]["rank"])
        if rank_val <= 20:   # already in top — not hidden
            continue
        if rank_val > 200:   # too low to be interesting
            continue
        q    = qualification_features(cand)
        b    = behavioral_features(cand)
        c    = contradiction_features(cand)
        risk = calculate_risk(cand)
        # Hidden gem criteria: strong semantic + strong behavior + low keyword overlap
        if q["semantic_score"] > 0.52 and b["response_rate"] > 0.65 and q["skill_overlap"] < 0.5 and c["contradiction_score"] == 0:
            gems.append({
                "candidate_id": cid,
                "rank": rank_val,
                "semantic": round(q["semantic_score"], 3),
                "behavioral": round(b["response_rate"], 2),
                "skill_overlap": round(q["skill_overlap"], 2),
                "fit": float(row.iloc[0]["score"]),
                "risk": risk,
                "title": cand["profile"].get("current_title", ""),
                "years": cand["profile"].get("years_of_experience", 0),
            })
    gems.sort(key=lambda x: x["semantic"] + x["behavioral"], reverse=True)
    return gems[:n]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:20px;font-weight:700;color:#3b82f6;letter-spacing:-0.5px;">⚡ EvidenceRank</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:24px;">Candidate Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:2px;color:#3b82f6;margin:20px 0 8px 0;">Job Profile</div>', unsafe_allow_html=True)
    st.markdown(f"**Role** &nbsp; `{jd['role_type']}`")
    st.markdown(f"**Level** &nbsp; `{jd['seniority']}`")
    st.markdown(f"**Org** &nbsp; `{jd['company_style']}`")
    st.markdown('<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:2px;color:#3b82f6;margin:20px 0 8px 0;">Required Skills</div>', unsafe_allow_html=True)
    skills_html = "".join([f'<span class="jd-pill">{s}</span>' for s in jd["required_skills"]])
    st.markdown(skills_html, unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:2px;color:#3b82f6;margin:20px 0 8px 0;">Dataset</div>', unsafe_allow_html=True)
    st.markdown("**Candidates scored** &nbsp; `100,000`")
    st.markdown("**Shortlist** &nbsp; `Top 100`")
    st.markdown("**Model** &nbsp; `MiniLM-L6-v2`")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-header"><div class="page-title">Candidate <span>Intelligence</span> Dashboard</div><div class="page-subtitle">Semantic fit · Behavioral signals · Contradiction detection · Evidence-based ranking</div></div>', unsafe_allow_html=True)

# ── Summary metrics ───────────────────────────────────────────────────────────
avg_score = df.head(100)["score"].mean()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value metric-accent">{df.iloc[0]["score"]:.3f}</div><div class="metric-label">Top Score</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_score:.3f}</div><div class="metric-label">Avg Top-100 Score</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value metric-accent">100</div><div class="metric-label">Shortlisted</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">100K</div><div class="metric-label">Candidates Ranked</div></div>', unsafe_allow_html=True)

# ── Decision Buckets ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Recruiter Decision View</div>', unsafe_allow_html=True)
st.markdown('<p style="font-size:13px;color:#475569;margin-bottom:16px;">Who to interview, who to review, who to skip — at a glance.</p>', unsafe_allow_html=True)

strong_hires, needs_review, high_risk = [], [], []
for _, row in df.head(100).iterrows():
    cand = candidate_map.get(row["candidate_id"])
    if not cand:
        continue
    c_feat = contradiction_features(cand)
    r_feat = calculate_risk(cand)
    fit    = float(row["score"])
    contras = int(c_feat["contradiction_score"])
    label = hire_label(fit, r_feat, contras)
    entry = {"id": row["candidate_id"], "fit": fit, "label": label}
    if "Strong Hire" in label:
        strong_hires.append(entry)
    elif "Needs Review" in label:
        needs_review.append(entry)
    else:
        high_risk.append(entry)

col_h, col_r, col_x = st.columns(3)

BUCKET_CARD = "background:#0d1220;border:1px solid #1e2d4a;border-radius:8px;padding:18px 20px;"

with col_h:
    rows_html = "".join([
        f'<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #1a2438;">'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:12px;color:#94a3b8;">{e["id"]}</span>'
        f'<span style="font-size:12px;font-weight:600;color:#4ade80;">{e["fit"]:.3f}</span></div>'
        for e in strong_hires[:8]
    ])
    st.markdown(
        f'<div style="{BUCKET_CARD}">'
        f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:2px;color:#4ade80;margin-bottom:12px;">🟢 Strong Hire ({len(strong_hires)})</div>'
        f'{rows_html}</div>',
        unsafe_allow_html=True
    )

with col_r:
    rows_html = "".join([
        f'<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #1a2438;">'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:12px;color:#94a3b8;">{e["id"]}</span>'
        f'<span style="font-size:12px;font-weight:600;color:#fbbf24;">{e["fit"]:.3f}</span></div>'
        for e in needs_review[:8]
    ])
    st.markdown(
        f'<div style="{BUCKET_CARD}">'
        f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:2px;color:#fbbf24;margin-bottom:12px;">🟡 Needs Review ({len(needs_review)})</div>'
        f'{rows_html}</div>',
        unsafe_allow_html=True
    )

with col_x:
    rows_html = "".join([
        f'<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #1a2438;">'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:12px;color:#94a3b8;">{e["id"]}</span>'
        f'<span style="font-size:12px;font-weight:600;color:#f87171;">{e["fit"]:.3f}</span></div>'
        for e in high_risk[:8]
    ]) if high_risk else '<div style="color:#475569;font-size:13px;padding:8px 0;">No high-risk candidates in top 100. All stable.</div>'
    st.markdown(
        f'<div style="{BUCKET_CARD}">'
        f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:2px;color:#f87171;margin-bottom:12px;">🔴 High Risk ({len(high_risk)})</div>'
        f'{rows_html}</div>',
        unsafe_allow_html=True
    )

# ── Hidden Gems ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Hidden Gems</div>', unsafe_allow_html=True)
st.markdown('<p style="font-size:13px;color:#475569;margin-bottom:16px;">Candidates with strong semantic fit and behavioral signals — missed by keyword filters.</p>', unsafe_allow_html=True)

with st.spinner("Finding hidden gems…"):
    gems = find_hidden_gems(candidate_map, df, n=5)

if gems:
    cols = st.columns(len(gems))
    for col, gem in zip(cols, gems):
        with col:
            st.markdown(
                f'<div class="gem-card">'
                f'<div class="gem-title">{gem["candidate_id"]}</div>'
                f'<div class="gem-sub">{gem["title"]} · {gem["years"]:.0f} yrs</div>'
                f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;">'
                f'<div style="background:#0f2040;color:#7dd3fc;border:1px solid #1e3a6a;border-radius:4px;padding:3px 8px;font-size:11px;">Semantic {gem["semantic"]:.2f}</div>'
                f'<div style="background:#0f2040;color:#7dd3fc;border:1px solid #1e3a6a;border-radius:4px;padding:3px 8px;font-size:11px;">Engage {gem["behavioral"]:.2f}</div>'
                f'<div style="background:#1e2d4a;color:#94a3b8;border:1px solid #2d3f5a;border-radius:4px;padding:3px 8px;font-size:11px;">Rank #{gem["rank"]}</div>'
                f'</div>'
                f'<div style="margin-top:8px;font-size:11px;color:#475569;">Low keyword overlap — would be missed by traditional ATS</div>'
                f'</div>',
                unsafe_allow_html=True
            )
else:
    st.markdown('<p style="color:#475569;font-size:13px;">No hidden gems found in current dataset.</p>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Top candidates table ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Top Ranked Candidates</div>', unsafe_allow_html=True)

table_rows = []
for _, row in df.head(20).iterrows():
    cand = candidate_map.get(row["candidate_id"])
    if not cand:
        continue
    c_feat  = contradiction_features(cand)
    r_score = calculate_risk(cand)
    fit = float(row["score"])
    contras = int(c_feat["contradiction_score"])

    label_html = hire_label(fit, r_score, contras)

    table_rows.append({
        "Rank": int(row["rank"]),
        "Candidate ID": row["candidate_id"],
        "Score": f"{fit:.4f}",
        "Decision": (
            "Strong Hire" if "Strong" in label_html
            else "Needs Review" if "Review" in label_html
            else "High Risk"
        ),
        "Reasoning": (
            str(row["reasoning"])[:80] + "…"
            if len(str(row["reasoning"])) > 80
            else str(row["reasoning"])
        ),
})

st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True, column_config={
    "Rank":     st.column_config.NumberColumn(width="small"),
    "Score":    st.column_config.TextColumn(width="small"),
    "Decision": st.column_config.TextColumn(width="medium"),
})

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Deep Candidate Analysis ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Deep Candidate Analysis</div>', unsafe_allow_html=True)
selected = st.selectbox("Select candidate to analyze", candidate_ids, label_visibility="collapsed")

candidate_row  = df[df["candidate_id"] == selected].iloc[0]
full_candidate = candidate_map[selected]
q    = qualification_features(full_candidate)
b    = behavioral_features(full_candidate)
c    = contradiction_features(full_candidate)
conf = confidence_features(full_candidate)
risk = calculate_risk(full_candidate)
why_review = []

signals = full_candidate.get("redrob_signals", {})

relocate = signals.get("willing_to_relocate", True)

if risk > 0.6:
    why_review.append("High notice period / stability risk")

if not relocate:
    why_review.append("Relocation concern")
if not relocate:
    why_review.append("Relocation concern")

fit            = float(candidate_row["score"])
rank_val       = int(candidate_row["rank"])
contradictions = int(c["contradiction_score"])
confidence_val = float(conf["evidence_density"])

# Decision badge
badge = hire_label(fit, risk, contradictions)

if badge == "Strong Hire":
    badge_html = '<span class="hire-badge-green">🟢 Strong Hire</span>'
elif badge == "High Risk":
    badge_html = '<span class="hire-badge-red">🔴 High Risk</span>'
else:
    badge_html = '<span class="hire-badge-yellow">🟡 Needs Review</span>'

st.markdown(
    f'<div style="margin-bottom:16px;">{badge_html}</div>',
    unsafe_allow_html=True
)

if badge == "Needs Review" and why_review:
    for reason in why_review:
        st.warning(reason)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value metric-accent">#{rank_val}</div><div class="metric-label">Rank</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{fit*100:.1f}%</div><div class="metric-label">Fit Score</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{confidence_val*100:.1f}%</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
with col4:
    acc = "metric-accent" if contradictions == 0 else ""
    st.markdown(f'<div class="metric-card"><div class="metric-value {acc}">{contradictions}</div><div class="metric-label">Contradictions</div></div>', unsafe_allow_html=True)
with col5:
    if risk < 0.35:
        label = "Low"
        color = "#4ade80"
    elif risk < 0.65:
        label = "Medium"
        color = "#fbbf24"
    else:
        label = "High"
        color = "#f87171"

    st.markdown(
        f'''
        <div class="metric-card">
            <div class="metric-value" style="color:{color};">{label}</div>
            <div class="metric-label">Risk Level</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f'<div class="reasoning-box">{candidate_row["reasoning"]}</div>', unsafe_allow_html=True)

# ── JD Skill Match ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
skill_match = skill_match_for(full_candidate)
match_html = ""
for skill, matched in skill_match.items():
    if matched:
        match_html += f'<div style="display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid #1a2438;"><span style="color:#4ade80;font-size:14px;">✅</span><span style="font-size:13px;color:#f1f5f9;">{skill}</span><span style="margin-left:auto;font-size:11px;color:#4ade80;">Matched</span></div>'
    else:
        match_html += f'<div style="display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid #1a2438;"><span style="color:#f87171;font-size:14px;">❌</span><span style="font-size:13px;color:#94a3b8;">{skill}</span><span style="margin-left:auto;font-size:11px;color:#f87171;">Missing</span></div>'

col_left, col_right = st.columns(2)

with col_left:
    st.markdown(
        f'<div style="background:#0d1220;border:1px solid #1e2d4a;border-radius:8px;padding:18px 20px;">'
        f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:2px;color:#3b82f6;margin-bottom:12px;">JD Skill Match</div>'
        f'{match_html}'
        f'</div>',
        unsafe_allow_html=True
    )

with col_right:
    st.markdown('<p style="font-size:12px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1.5px;">Candidate Profile</p>', unsafe_allow_html=True)
    r_vals = [
        min(1.0, q["semantic_score"]),
        max(0.05, q["title_score"]),
        max(0.05, confidence_val),
        max(0.05, 1 - risk),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=r_vals,
        theta=["Semantic Fit", "Seniority", "Confidence", "Low Risk"],
        fill='toself',
        fillcolor='rgba(59,130,246,0.15)',
        line=dict(color='#3b82f6', width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1220",
            radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e2d4a", linecolor="#1e2d4a", tickfont=dict(color="#475569", size=9)),
            angularaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a", tickfont=dict(color="#94a3b8", size=11)),
        ),
        paper_bgcolor="#080c14", plot_bgcolor="#080c14",
        font=dict(color="#94a3b8", family="Inter"),
        showlegend=False, height=300, margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# Score breakdown
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p style="font-size:12px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1.5px;">Score Breakdown</p>', unsafe_allow_html=True)
qual    = round(q["semantic_score"]*0.35 + q["skill_overlap"]*0.20 + q["title_score"]*0.20, 3)
beh     = round(b["github_score"]*0.35 + b["response_rate"]*0.25 + b["completion_rate"]*0.20 + b["recruiter_saves"]*0.20, 3)
penalty = round(min(c["contradiction_score"]/5, 1)*0.25, 3)
names  = ["Semantic Fit", "Behavioral", "Confidence"]
vals   = [qual, beh, confidence_val]
clrs   = ["#3b82f6", "#22d3ee", "#a78bfa"]
if penalty > 0:
    names.append("Penalty"); vals.append(-penalty); clrs.append("#f87171")

fig2 = go.Figure()
for name, val, color in zip(names, vals, clrs):
    fig2.add_trace(go.Bar(x=[name], y=[val], marker_color=color, showlegend=False,
        text=[f"{val:.3f}"], textposition="outside", textfont=dict(color="#94a3b8", size=11)))
fig2.update_layout(
    paper_bgcolor="#080c14", plot_bgcolor="#0d1220",
    font=dict(color="#94a3b8", family="Inter"),
    height=280, margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a", tickfont=dict(color="#94a3b8")),
    yaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a", zeroline=True, zerolinecolor="#2d3250", tickfont=dict(color="#475569")),
    bargap=0.35,
)
st.plotly_chart(fig2, use_container_width=True)

# ── Strengths / Contradictions / Risk ─────────────────────────────────────────
CARD_STYLE  = "background:#0d1220;border:1px solid #1e2d4a;border-radius:8px;padding:18px 20px;min-height:150px;"
LABEL_STYLE = "font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:2px;color:#3b82f6;margin-bottom:14px;"

col_a, col_b, col_c = st.columns(3)
with col_a:
    strengths = []
    title_lower = full_candidate["profile"].get("current_title", "").lower()
    if "staff" in title_lower or "principal" in title_lower:
        strengths.append("Staff-level seniority match")
    if q["semantic_score"] > 0.55:
        strengths.append("Strong semantic JD alignment")
    if b["response_rate"] > 0.7:
        strengths.append("High recruiter engagement")
    if b["github_score"] > 0.6:
        strengths.append("Active GitHub presence")
    if fit > 0.70:
        strengths.append("Top-tier overall fit")
    if not strengths:
        strengths.append("Solid generalist profile")
    tags_html = "".join([tag(TAG_GREEN, "✓", s) for s in strengths])
    st.markdown(f'<div style="{CARD_STYLE}"><div style="{LABEL_STYLE}">Strengths</div>{tags_html}</div>', unsafe_allow_html=True)

with col_b:
    inner = tag(TAG_GREEN, "✓", "No contradictions detected") if contradictions == 0 else "".join([tag(TAG_RED, "⚠", r) for r in c["reasons"]])
    st.markdown(f'<div style="{CARD_STYLE}"><div style="{LABEL_STYLE}">Contradictions</div>{inner}</div>', unsafe_allow_html=True)

with col_c:
    risks = []
    if fit < 0.65: risks.append("Below-average fit score")
    if confidence_val < 0.5: risks.append("Low evidence density")
    if contradictions >= 2: risks.append("Multiple signal conflicts")
    if risk > 0.6: risks.append("High notice / relocation risk")
    inner_r = tag(TAG_GREEN, "✓", "Low overall risk") if not risks else "".join([tag(TAG_ORANGE, "△", r) for r in risks])
    st.markdown(f'<div style="{CARD_STYLE}"><div style="{LABEL_STYLE}">Risk Signals</div>{inner_r}</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Candidate Comparison ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Candidate Comparison</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    compare1 = st.selectbox("Candidate A", candidate_ids, index=0, key="c1")
with col2:
    compare2 = st.selectbox("Candidate B", candidate_ids, index=1, key="c2")

cand_a = df[df["candidate_id"] == compare1].iloc[0]
cand_b = df[df["candidate_id"] == compare2].iloc[0]
fa, fb = candidate_map[compare1], candidate_map[compare2]
qa, qb       = qualification_features(fa), qualification_features(fb)
ba, bb       = behavioral_features(fa), behavioral_features(fb)
ca, cb       = contradiction_features(fa), contradiction_features(fb)
confa, confb = confidence_features(fa), confidence_features(fb)

metrics_data = {
    "Metric": ["Rank","Fit Score","Semantic Match","Skill Overlap","Response Rate","GitHub Score","Confidence","Contradictions"],
    compare1: [int(cand_a["rank"]), f"{float(cand_a['score'])*100:.1f}%", f"{qa['semantic_score']:.3f}", f"{qa['skill_overlap']:.3f}", f"{ba['response_rate']:.2f}", f"{ba['github_score']:.2f}", f"{confa['evidence_density']*100:.1f}%", int(ca["contradiction_score"])],
    compare2: [int(cand_b["rank"]), f"{float(cand_b['score'])*100:.1f}%", f"{qb['semantic_score']:.3f}", f"{qb['skill_overlap']:.3f}", f"{bb['response_rate']:.2f}", f"{bb['github_score']:.2f}", f"{confb['evidence_density']*100:.1f}%", int(cb["contradiction_score"])],
}
st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

winner = compare1 if float(cand_a["score"]) >= float(cand_b["score"]) else compare2
reasons = []
if abs(float(cand_a["score"]) - float(cand_b["score"])) * 100 > 0.01:
    reasons.append(f"Higher fit score by {abs(float(cand_a['score'])-float(cand_b['score']))*100:.2f}%")
if (compare1==winner and qa["semantic_score"]>qb["semantic_score"]) or (compare2==winner and qb["semantic_score"]>qa["semantic_score"]):
    reasons.append("Stronger semantic alignment with JD")
if (compare1==winner and ba["response_rate"]>bb["response_rate"]) or (compare2==winner and bb["response_rate"]>ba["response_rate"]):
    reasons.append("Higher recruiter engagement signal")
if (compare1==winner and ca["contradiction_score"]<cb["contradiction_score"]) or (compare2==winner and cb["contradiction_score"]<ca["contradiction_score"]):
    reasons.append("Fewer profile contradictions")
if reasons:
    st.markdown(f'<div class="reasoning-box" style="margin-top:16px;"><strong>{winner}</strong> ranks higher: {" · ".join(reasons)}</div>', unsafe_allow_html=True)

# ── Why EvidenceRank Wins ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">EvidenceRank vs Traditional ATS</div>', unsafe_allow_html=True)
st.markdown('<p style="font-size:13px;color:#475569;margin-bottom:20px;">Why keyword filters miss the best candidates — and how EvidenceRank finds them.</p>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    comparison_rows = [
        ("Matching method",      "Keyword overlap",         "Semantic embeddings"),
        ("Behavioral signals",   "❌ Ignored",              "✅ GitHub, response rate, saves"),
        ("Contradiction check",  "❌ None",                 "✅ 5 rule-based detectors"),
        ("Hidden gems",          "❌ Missed",               "✅ Surfaced explicitly"),
        ("Explainability",       "❌ Black box score",      "✅ Per-candidate reasoning"),
        ("Confidence scoring",   "❌ Not available",        "✅ Evidence density metric"),
    ]
    header = (
        '<div style="background:#0d1220;border:1px solid #1e2d4a;border-radius:8px;overflow:hidden;">'
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;background:#111827;padding:10px 16px;">'
        '<span style="font-size:11px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:1px;">Signal</span>'
        '<span style="font-size:11px;font-weight:600;color:#f87171;text-transform:uppercase;letter-spacing:1px;">Keyword ATS</span>'
        '<span style="font-size:11px;font-weight:600;color:#4ade80;text-transform:uppercase;letter-spacing:1px;">EvidenceRank</span>'
        '</div>'
    )
    body = ""
    for label, ats, er in comparison_rows:
        body += (
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;padding:10px 16px;border-top:1px solid #1a2438;">'
            f'<span style="font-size:12px;color:#94a3b8;">{label}</span>'
            f'<span style="font-size:12px;color:#f87171;">{ats}</span>'
            f'<span style="font-size:12px;color:#4ade80;">{er}</span>'
            f'</div>'
        )
    st.markdown(header + body + '</div>', unsafe_allow_html=True)

with col_r:
    # Bar chart: how many top-20 are semantic-only vs keyword-only discoveries
    top20_ids = df.head(20)["candidate_id"].tolist()
    semantic_only, keyword_match, both = 0, 0, 0
    for cid in top20_ids:
        cand = candidate_map.get(cid)
        if not cand:
            continue
        q_feat = qualification_features(cand)
        if q_feat["skill_overlap"] >= 0.5 and q_feat["semantic_score"] >= 0.52:
            both += 1
        elif q_feat["semantic_score"] >= 0.52:
            semantic_only += 1
        else:
            keyword_match += 1

    fig3 = go.Figure(go.Bar(
        x=["Keyword + Semantic", "Semantic Only\n(would be missed)", "Keyword Only"],
        y=[both, semantic_only, keyword_match],
        marker_color=["#3b82f6", "#4ade80", "#f87171"],
        text=[both, semantic_only, keyword_match],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=13),
    ))
    fig3.update_layout(
        title=dict(text="Top-20 Candidates: How They Were Found", font=dict(color="#94a3b8", size=13), x=0),
        paper_bgcolor="#080c14", plot_bgcolor="#0d1220",
        font=dict(color="#94a3b8", family="Inter"),
        height=300, margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a", tickfont=dict(color="#94a3b8", size=11)),
        yaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a", tickfont=dict(color="#475569")),
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        '<div style="background:#052e16;border:1px solid #166534;border-radius:6px;padding:10px 14px;font-size:12px;color:#4ade80;">'
        f'✅ <strong>{semantic_only}</strong> of your top 20 candidates would be <strong>invisible to keyword ATS</strong> — only found through semantic understanding.'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;font-size:11px;color:#1e2d4a;letter-spacing:2px;text-transform:uppercase;">EvidenceRank © 2026 · Semantic Ranking · Signal Intelligence · Trust-Aware Scoring</div>', unsafe_allow_html=True)