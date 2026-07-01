from docx import Document
import re


def extract_text(docx_path):
    doc = Document(docx_path)

    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())

    return "\n".join(text)
def extract_job_title(text):
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if 3 < len(line) < 100:
            return line

    return "Unknown Role"


def parse_job_description(docx_path):
    text = extract_text(docx_path)
    lower_text = text.lower()
    job_title = extract_job_title(text)

    skill_keywords = [
        # AI
        "python", "machine learning", "deep learning", "nlp",
        "rag", "llm", "embeddings", "vector database",
        "retrieval", "ranking", "pytorch", "tensorflow",

        # Backend
        "sql", "postgresql", "mysql", "redis", "kafka",
        "aws", "gcp", "azure", "docker", "kubernetes",
        "microservices", "distributed systems",
        "api", "rest", "fastapi", "django",

        # Frontend
        "react", "javascript", "typescript",

        # Marketing
        "seo", "sem", "google analytics",
        "content marketing", "campaign management",

        # Sales
        "crm", "lead generation",
        "negotiation", "sales strategy"
    ]

    found_skills = []

    for skill in skill_keywords:
        if skill in lower_text:
            found_skills.append(skill)

    seniority = "mid"

    if "staff" in lower_text or "principal" in lower_text:
        seniority = "staff"
    elif "senior" in lower_text:
        seniority = "senior"
    elif "intern" in lower_text:
        seniority = "intern"

    

    company_style = "enterprise"

    if "startup" in lower_text:
        company_style = "startup"

    return {
        "job_title": job_title,
        "required_skills": found_skills,
        "seniority": seniority,
        "company_style": company_style
    }
def extract_job_title(text):
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if len(line) > 3 and len(line) < 100:
            return line

    return "Unknown Role"


if __name__ == "__main__":
    jd = parse_job_description("../data/raw/job_description.docx")
    print("Job title:", jd["job_title"])
    print("Skills:", jd["required_skills"])
    print("Seniority:", jd["seniority"])