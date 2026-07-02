# ⚡ EvidenceRank — AI-Powered Candidate Intelligence & Ranking

EvidenceRank is an AI-driven hiring intelligence system that ranks candidates the way an experienced recruiter would — not through simple keyword matching, but through semantic understanding, behavioral analysis, contradiction detection, and trust-aware scoring.

## Problem Statement

Recruiters go through thousands of profiles and often miss high-quality candidates because traditional Applicant Tracking Systems (ATS) rely heavily on keyword matching.

This causes major problems:
- Strong candidates get filtered out due to missing exact keywords
- Hidden talent remains undiscovered
- Recruiters spend excessive time manually screening profiles
- Hiring decisions become noisy and inconsistent

EvidenceRank solves this by ranking candidates based on **actual relevance**, not keyword overlap alone.

---

## Key Features

### 🧠 Semantic Candidate Matching
Uses sentence embeddings to understand the meaning of job descriptions and candidate profiles.

Instead of:
> “Does resume contain exact skill keyword?”

EvidenceRank asks:
> “Does this candidate truly fit the role?”

---

### 📄 Intelligent Job Description Parsing
Automatically extracts:
- Role type
- Seniority level
- Required skills
- Organization style
- Role family

Example:
```json
{
  "role_type": "ml",
  "seniority": "principal",
  "required_skills": ["python", "llm", "rag", "nlp", "rest"]
}
```

---

### 📊 Hybrid Candidate Scoring

Each candidate receives a final score using multiple signals:

#### Qualification Signals
- Semantic similarity
- Skill overlap
- Title relevance
- Experience match

#### Behavioral Signals
- Recruiter engagement
- GitHub activity
- Response rate
- Platform activity

#### Trust Signals
- Contradiction detection
- Confidence scoring
- Risk assessment

---

### 🚨 Contradiction Detection
EvidenceRank detects suspicious profile inconsistencies such as:
- Senior title with very low experience
- Unrealistic skill combinations
- Missing supporting evidence
- Risky behavioral signals

---

### 💎 Hidden Gem Discovery
Traditional ATS often misses candidates with low keyword overlap but strong semantic relevance.

EvidenceRank explicitly surfaces these “hidden gems.”

---

### 📈 Explainable AI Ranking
Each recommendation includes reasoning:

Example:
> Strong semantic alignment with job description; High required skill coverage; Strong recruiter engagement signals; Low hiring risk

This makes the system transparent and recruiter-friendly.

---

# Architecture

```text
Job Description
      ↓
JD Parser
      ↓
Feature Engineering
      ↓
Semantic Embeddings
      ↓
Hybrid Scoring Engine
      ↓
Candidate Ranking
      ↓
Dashboard + Explainability
```

---

# Tech Stack

## AI / ML
- Sentence Transformers (MiniLM-L6-v2)
- Semantic Embeddings
- Vector Similarity Ranking

## Backend
- Python
- Pandas
- NumPy

## Dashboard
- Streamlit
- Plotly

## Parsing
- Rule-based NLP parser
- Skill ontology

---

# Scoring Formula

Final fit score combines:

```text
Fit Score =
40% Qualification Score +
25% Behavioral Score +
20% Confidence Score -
15% Contradiction Penalty
```

Risk score is calculated separately for recruiter decision support.

---

# Project Structure

```bash
EvidenceRank/
│
├── data/
│   └── raw/
│
├── outputs/
│   └── final_submission.csv
│
├── src/
│   ├── dashboard.py
│   ├── features.py
│   ├── jd_parser.py
│   ├── rank_candidates.py
│   └── scorer.py
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone repository:

```bash
git clone https://github.com/Manvitha-Kopela/EvidenceRank.git
cd EvidenceRank
```

Create virtual environment:

```bash
python -m venv venv
```

Activate:

### Windows
```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Candidate Ranking

```bash
cd src
python rank_candidates.py ..\data\raw\job_description.docx
```

This generates:
- ranked_candidates.csv
- top_100_candidates.csv

---

# Running Dashboard

```bash
streamlit run src/dashboard.py
```

Dashboard includes:
- Recruiter decision buckets
- Hidden gems
- Candidate comparison
- Deep candidate analysis
- ATS vs EvidenceRank comparison

---

# Output Format

Submission file contains:

| candidate_id | rank | score | reasoning |
|-------------|------|-------|-----------|

---

# Why EvidenceRank Wins Over Traditional ATS

| Feature | Traditional ATS | EvidenceRank |
|---------|-----------------|--------------|
| Matching | Keyword overlap | Semantic understanding |
| Hidden gems | Missed | Found |
| Behavioral analysis | No | Yes |
| Explainability | Low | High |
| Trust signals | None | Yes |

---

# Impact

EvidenceRank helps recruiters:
- Reduce manual screening time
- Improve shortlist quality
- Discover hidden talent
- Make explainable hiring decisions

---

# Future Improvements

- LLM-based resume summarization
- Bias detection
- Adaptive scoring weights
- Interview question generation
- Real-time recruiter feedback loop

---

# Team

Built for **The Data & AI Challenge 2026**  
Team: **SkillSetu**

---

## Final Note

EvidenceRank transforms candidate ranking from **keyword filtering** into **intelligent evidence-based hiring**.

**Hire smarter. Hire with evidence.**