"""
scorer.py
Computes a relevance score for each resume against a job description.

Design choice (see TRADEOFFS.md for the full reasoning):
We use TF-IDF + cosine similarity rather than embeddings. It's fast, needs no
external model download or API call, and is fully explainable — you can point
at the exact overlapping terms. The tradeoff is it's a bag-of-words method: it
won't catch that "led a team" and "managed engineers" mean the same thing.
That's an explicit, documented limitation, not an oversight.

Final score is a weighted blend of:
  - TF-IDF cosine similarity (0-100)   weight 0.5
  - Skill keyword overlap (0-100)      weight 0.3
  - Experience match (0-100)           weight 0.15
  - Education match (0-100)            weight 0.05
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .extractor import (
    extract_skills,
    extract_years_experience,
    extract_education_level,
    
    build_skill_vocab_from_jd,
)

WEIGHTS = {
    "similarity": 0.45,
    "skills": 0.35,
    "experience": 0.15,
    "education": 0.05,
}

EDUCATION_RANK = {"unknown": 0, "bachelor": 1, "master": 2, "phd": 3}


def compute_tfidf_similarity(jd_text: str, resume_texts: list) -> list:
    """Return cosine similarity (0-100) between the JD and each resume."""
    corpus = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    sims = cosine_similarity(jd_vector, resume_vectors)[0]
    return [round(float(s) * 100, 2) for s in sims]


def score_candidate(jd_text: str, jd_skills: set, jd_min_years: float,
                     jd_education: str, resume_text: str) -> dict:
    """Score a single resume against the JD on skills/experience/education.
    TF-IDF similarity is computed separately in batch (see score_all) since
    it needs the whole corpus at once.
    """
    resume_skills = extract_skills(resume_text)
    overlap = jd_skills & resume_skills
    skill_score = (len(overlap) / len(jd_skills) * 100) if jd_skills else 0.0

    years = extract_years_experience(resume_text)
    if jd_min_years <= 0:
        experience_score = 100.0
    else:
        experience_score = min(100.0, (years / jd_min_years) * 100)

    edu_level = extract_education_level(resume_text)
    jd_rank = EDUCATION_RANK.get(jd_education, 0)
    resume_rank = EDUCATION_RANK.get(edu_level, 0)
    education_score = 100.0 if resume_rank >= jd_rank else (resume_rank / jd_rank * 100 if jd_rank else 100.0)

    return {
        "matched_skills": sorted(overlap),
        "missing_skills": sorted(jd_skills - resume_skills),
        "skill_score": round(skill_score, 1),
        "years_experience": years,
        "experience_score": round(experience_score, 1),
        "education_level": edu_level,
        "education_score": round(education_score, 1),
    }


def score_all(jd_text: str, resumes: dict, jd_min_years: float = 0.0,
              jd_education: str = "unknown") -> list:
    """Score and rank every resume in `resumes` ({filename: text}) against jd_text.

    Returns a list of dicts sorted by total_score descending.
    """
    filenames = list(resumes.keys())
    resume_texts = [resumes[f] for f in filenames]

    jd_skill_vocab = build_skill_vocab_from_jd(jd_text)
    jd_skills = extract_skills(jd_text, vocab=jd_skill_vocab)

    similarities = compute_tfidf_similarity(jd_text, resume_texts)

    results = []
    for fname, resume_text, sim in zip(filenames, resume_texts, similarities):
        breakdown = score_candidate(jd_text, jd_skills, jd_min_years, jd_education, resume_text)
        total = (
            sim * WEIGHTS["similarity"]
            + breakdown["skill_score"] * WEIGHTS["skills"]
            + breakdown["experience_score"] * WEIGHTS["experience"]
            + breakdown["education_score"] * WEIGHTS["education"]
        )
        results.append({
            "filename": fname,
            "total_score": round(total, 1),
            "similarity_score": sim,
            **breakdown,
        })

    results.sort(key=lambda r: r["total_score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i
    return results
