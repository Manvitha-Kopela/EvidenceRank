import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jd_parser import parse_job_description
from features import contradiction_features, confidence_features, qualification_features, behavioral_features, load_embedding_cache
from parser import load_candidates
from scorer import calculate_scores, calculate_risk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "candidates.jsonl")
JD_PATH = os.path.join(BASE_DIR, "data", "raw", "job_description.docx")
RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "final_submission.csv")

st.set_page_config(
    page_title="EvidenceRank",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }

.stApp {
    background: #080c14;
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1e2d4a !important;
}

section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}

.sidebar-brand {
    font-size: 20px;
    font-weight: 700;
    color: #3b82f6 !important;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}

.sidebar-tagline {
    font-size: 11px;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 24px;
}

.sidebar-section-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #3b82f6 !important;
    margin: 20px 0 8px 0;
}

.jd-pill {
    display: inline-block;
    background: #1e2d4a;
    color: #7dd3fc !important;
    border: 1px solid #1e40af;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px 2px;
}

.page-header {
    padding: 32px 0 24px 0;
    border-bottom: 1px solid #1e2d4a;
    margin-bottom: 32px;
}

.page-title {
    font-size: 32px;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -1px;
    line-height: 1.1;
}

.page-title span {
    color: #3b82f6;
}

.page-subtitle {
    font-size: 14px;
    color: #475569;
    margin-top: 6px;
}

.section-header {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #3b82f6;
    margin: 40px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2d4a;
}

.metric-card {
    background: #0d1220;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}

.metric-label {
    font-size: 11px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
}

.metric-accent {
    color: #3b82f6;
}

.reasoning-box {
    background: #0d1220;
    border: 1px solid #1e2d4a;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    font-size: 14px;
    color: #94a3b8;
    line-height: 1.7;
    font-family: 'Inter', sans-serif;
}

.tag-good {
    display: inline-block;
    background: #052e16;
    color: #4ade80;
    border: 1px solid #166534;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 12px;
    margin: 4px 4px 4px 0;
}

.tag-warn {
    display: inline-block;
    background: #1c1917;
    color: #fb923c;
    border: 1px solid #7c2d12;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 12px;
    margin: 4px 4px 4px 0;
}

.tag-bad {
    display: inline-block;
    background: #1c0a0a;
    color: #f87171;
    border: 1px solid #7f1d1d;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 12px;
    margin: 4px 4px 4px 0;
}

.candidate-row {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid #1e2d4a;
    transition: background 0.15s;
}

.candidate-row:hover {
    background: #0d1220;
}

.rank-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: #475569;
    width: 32px;
    text-align: center;
}

.rank-badge.top3 {
    color: #3b82f6;
}

div[data-testid="stSelectbox"] > div {
    background: #0d1220 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 6px !important;
    color: #f1f5f9 !important;
}

.stDataFrame {
    background: transparent !important;
}

.stDataFrame [data-testid="stDataFrameResizable"] {
    background: #0d1220 !important;
}

.divider {
    border: none;
    border-top: 1px solid #1e2d4a;
    margin: 32px 0;
}
.stDataFrame > div {
    background: #0d1220 !important;
}
iframe {
    background: #0d1220 !important;
}
</style>
""", unsafe_allow_html=True)


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

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚡ EvidenceRank</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Candidate Intelligence</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Job Profile</div>', unsafe_allow_html=True)
    st.markdown(f"**Role** &nbsp; `{jd['role_type']}`")
    st.markdown(f"**Level** &nbsp; `{jd['seniority']}`")
    st.markdown(f"**Org** &nbsp; `{jd['company_style']}`")

    st.markdown('<div class="sidebar-section-label">Required Skills</div>', unsafe_allow_html=True)
    skills_html = "".join([f'<span class="jd-pill">{s}</span>' for s in jd["required_skills"]])
    st.markdown(skills_html, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown(f"**Candidates scored** &nbsp; `100,000`")
    st.markdown(f"**Shortlist** &nbsp; `Top 100`")
    st.markdown(f"**Model** &nbsp; `MiniLM-L6-v2`")

# Main content
st.markdown('<div class="page-header"><div class="page-title">Candidate <span>Intelligence</span> Dashboard</div><div class="page-subtitle">Semantic fit · Behavioral signals · Contradiction detection · Evidence-based ranking</div></div>', unsafe_allow_html=True)

# Summary metrics
top10 = df.head(10)
avg_score = df.head(100)["score"].mean()
zero_contradictions = 100  # all top-100 passed contradiction filter

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value metric-accent">{df.iloc[0]["score"]:.3f}</div><div class="metric-label">Top Score</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_score:.3f}</div><div class="metric-label">Avg Top-100 Score</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value metric-accent">100</div><div class="metric-label">Shortlisted</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">100K</div><div class="metric-label">Candidates Ranked</div></div>', unsafe_allow_html=True)

# Top candidates table
st.markdown('<div class="section-header">Top Ranked Candidates</div>', unsafe_allow_html=True)

display_df = df[["rank", "candidate_id", "score", "reasoning"]].head(20).copy()
display_df.columns = ["Rank", "Candidate ID", "Score", "Reasoning"]
display_df["Reasoning"] = display_df["Reasoning"].apply(lambda x: x[:80] + "..." if len(str(x)) > 80 else x)
display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.4f}")
display_df["Reasoning"] = display_df["Reasoning"].apply(lambda x: x[:80] + "..." if len(str(x)) > 80 else x)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "Score": st.column_config.TextColumn(width="small"),
    }
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Candidate Deep Analysis
st.markdown('<div class="section-header">Deep Candidate Analysis</div>', unsafe_allow_html=True)

selected = st.selectbox("Select candidate to analyze", candidate_ids, label_visibility="collapsed")

candidate_row = df[df["candidate_id"] == selected].iloc[0]
full_candidate = candidate_map[selected]

q = qualification_features(full_candidate)
b = behavioral_features(full_candidate)
c = contradiction_features(full_candidate)
conf = confidence_features(full_candidate)
scores = calculate_scores(q, b, c, conf)
risk = calculate_risk(full_candidate)

fit = float(candidate_row["score"])
rank = int(candidate_row["rank"])
contradictions = int(c["contradiction_score"])
confidence_val = float(conf["evidence_density"])

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value metric-accent">#{rank}</div><div class="metric-label">Rank</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{fit*100:.1f}%</div><div class="metric-label">Fit Score</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{confidence_val*100:.1f}%</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
with col4:
    color = "metric-accent" if contradictions == 0 else ""
    st.markdown(f'<div class="metric-card"><div class="metric-value {color}">{contradictions}</div><div class="metric-label">Contradictions</div></div>', unsafe_allow_html=True)
with col5:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{risk*100:.1f}%</div><div class="metric-label">Risk Score</div></div>', unsafe_allow_html=True)

# Reasoning
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f'<div class="reasoning-box">{candidate_row["reasoning"]}</div>', unsafe_allow_html=True)

# Charts row
st.markdown("<br>", unsafe_allow_html=True)
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Candidate Profile**")
    skill_score = min(1.0, q["semantic_score"])
    seniority_score = q["title_score"]
    conf_score = confidence_val
    risk_score = max(0, 1 - risk)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[skill_score, seniority_score, conf_score, risk_score, skill_score],
        theta=["Semantic Fit", "Seniority", "Confidence", "Low Risk", "Semantic Fit"],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(color='#3b82f6', width=2),
        name='Candidate'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1220",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e2d4a", linecolor="#1e2d4a", tickfont=dict(color="#475569", size=9)),
            angularaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a", tickfont=dict(color="#94a3b8", size=11))
        ),
        paper_bgcolor="#080c14",
        plot_bgcolor="#080c14",
        font=dict(color="#94a3b8", family="Inter"),
        showlegend=False,
        height=320,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("**Score Breakdown**")
    qual = round(q["semantic_score"] * 0.35 + q["skill_overlap"] * 0.20 + q["title_score"] * 0.20, 3)
    beh = round(b["github_score"] * 0.35 + b["response_rate"] * 0.25 + b["completion_rate"] * 0.20 + b["recruiter_saves"] * 0.20, 3)
    penalty = round(min(c["contradiction_score"] / 5, 1) * 0.25, 3)

    breakdown_data = {
        "Component": ["Semantic Fit", "Behavioral", "Confidence", "Penalty"],
        "Value": [qual, beh, confidence_val, -penalty],
        "Color": ["#3b82f6", "#22d3ee", "#a78bfa", "#f87171"]
    }

    fig2 = go.Figure()
    for i, row in enumerate(zip(breakdown_data["Component"], breakdown_data["Value"], breakdown_data["Color"])):
        name, val, color = row
        fig2.add_trace(go.Bar(
            x=[name], y=[val],
            marker_color=color,
            name=name,
            showlegend=False
        ))

    fig2.update_layout(
        paper_bgcolor="#080c14",
        plot_bgcolor="#0d1220",
        font=dict(color="#94a3b8", family="Inter"),
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a"),
        yaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a"),
        bargap=0.3
    )
    st.plotly_chart(fig2, use_container_width=True)

# Strengths / Contradictions / Risk
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**Strengths**")
    strengths = []
    if "staff" in full_candidate["profile"].get("current_title", "").lower():
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
    for s in strengths:
        st.markdown(f'<span class="tag-good">✓ {s}</span>', unsafe_allow_html=True)

with col_b:
    st.markdown("**Contradictions**")
    if contradictions == 0:
        st.markdown('<span class="tag-good">✓ No contradictions detected</span>', unsafe_allow_html=True)
    else:
        for reason in c["reasons"]:
            st.markdown(f'<span class="tag-bad">⚠ {reason}</span>', unsafe_allow_html=True)

with col_c:
    st.markdown("**Risk Signals**")
    risks = []
    if fit < 0.65:
        risks.append("Below-average fit score")
    if confidence_val < 0.5:
        risks.append("Low evidence density")
    if contradictions >= 2:
        risks.append("Multiple signal conflicts")
    if risk > 0.6:
        risks.append("High notice period / relocation risk")
    if not risks:
        risks.append("Low overall risk")
        for r in risks:
            st.markdown(f'<span class="tag-good">✓ {r}</span>', unsafe_allow_html=True)
    else:
        for r in risks:
            st.markdown(f'<span class="tag-warn">△ {r}</span>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Candidate Comparison
st.markdown('<div class="section-header">Candidate Comparison</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    compare1 = st.selectbox("Candidate A", candidate_ids, index=0, key="c1")
with col2:
    compare2 = st.selectbox("Candidate B", candidate_ids, index=1, key="c2")

cand_a = df[df["candidate_id"] == compare1].iloc[0]
cand_b = df[df["candidate_id"] == compare2].iloc[0]

fa = candidate_map[compare1]
fb = candidate_map[compare2]

qa = qualification_features(fa)
qb = qualification_features(fb)
ba = behavioral_features(fa)
bb = behavioral_features(fb)
ca = contradiction_features(fa)
cb = contradiction_features(fb)
confa = confidence_features(fa)
confb = confidence_features(fb)

metrics_data = {
    "Metric": ["Rank", "Fit Score", "Semantic Match", "Skill Overlap", "Response Rate", "GitHub Score", "Confidence", "Contradictions"],
    compare1: [
        int(cand_a["rank"]),
        f"{float(cand_a['score'])*100:.1f}%",
        f"{qa['semantic_score']:.3f}",
        f"{qa['skill_overlap']:.3f}",
        f"{ba['response_rate']:.2f}",
        f"{ba['github_score']:.2f}",
        f"{confa['evidence_density']*100:.1f}%",
        int(ca["contradiction_score"])
    ],
    compare2: [
        int(cand_b["rank"]),
        f"{float(cand_b['score'])*100:.1f}%",
        f"{qb['semantic_score']:.3f}",
        f"{qb['skill_overlap']:.3f}",
        f"{bb['response_rate']:.2f}",
        f"{bb['github_score']:.2f}",
        f"{confb['evidence_density']*100:.1f}%",
        int(cb["contradiction_score"])
    ]
}

st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

winner = compare1 if float(cand_a["score"]) >= float(cand_b["score"]) else compare2
loser = compare2 if winner == compare1 else compare1

reasons = []
score_diff = abs(float(cand_a["score"]) - float(cand_b["score"])) * 100
if score_diff > 0.01:
    reasons.append(f"Higher fit score by {score_diff:.2f}%")
if (compare1 == winner and qa["semantic_score"] > qb["semantic_score"]) or (compare2 == winner and qb["semantic_score"] > qa["semantic_score"]):
    reasons.append("Stronger semantic alignment with JD")
if (compare1 == winner and ba["response_rate"] > bb["response_rate"]) or (compare2 == winner and bb["response_rate"] > ba["response_rate"]):
    reasons.append("Higher recruiter engagement signal")
if (compare1 == winner and ca["contradiction_score"] < cb["contradiction_score"]) or (compare2 == winner and cb["contradiction_score"] < ca["contradiction_score"]):
    reasons.append("Fewer profile contradictions")

if reasons:
    summary = f"<strong>{winner}</strong> ranks higher: " + " · ".join(reasons)
    st.markdown(f'<div class="reasoning-box">{summary}</div>', unsafe_allow_html=True)
st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;font-size:11px;color:#1e2d4a;letter-spacing:2px;text-transform:uppercase;">EvidenceRank © 2026 · Semantic Ranking · Signal Intelligence · Trust-Aware Scoring</div>', unsafe_allow_html=True)