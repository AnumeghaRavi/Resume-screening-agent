"""
extractor.py
Pulls structured, checkable signals out of raw resume/JD text:
  - years of experience
  - education level
  - skill keyword hits

These are used as an *explainable* layer on top of the TF-IDF similarity score
in scorer.py, so a reviewer can see exactly why a candidate scored the way
they did, not just a black-box number.
"""

import re

# A reasonably broad default vocabulary for software/data roles. In a real
# deployment this would be swapped for a role-specific taxonomy or pulled
# dynamically from the JD itself (see build_skill_vocab_from_jd below).
DEFAULT_SKILL_VOCAB = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "nosql", "postgresql", "mysql", "mongodb", "redis",
    "django", "flask", "fastapi", "react", "node.js", "next.js",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
    "git", "rest api", "graphql", "microservices",
    "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow",
    "machine learning", "deep learning", "nlp", "data analysis",
    "agile", "scrum", "jira",
    "html", "css", "linux", "bash",
]

EDUCATION_LEVELS = [
    ("phd", ["phd", "ph.d", "doctorate"]),
    ("master", ["msc", "m.sc", "master", "mba", "m.tech", "mtech"]),
    ("bachelor", ["bsc", "b.sc", "bachelor", "b.tech", "btech", "be ", "b.e."]),
]


def extract_skills(text: str, vocab=None) -> set:
    """Return the set of vocabulary skills that appear in text (case-insensitive)."""
    vocab = vocab or DEFAULT_SKILL_VOCAB
    text_lower = text.lower()
    found = set()
    for skill in vocab:
        # word-boundary-ish match so "go" doesn't match inside "google"
        pattern = r"(?<![a-zA-Z])" + re.escape(skill.lower()) + r"(?![a-zA-Z])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def extract_years_experience(text: str) -> float:
    """Best-effort extraction of total years of experience.

    Looks for explicit phrases like "5 years of experience" first. Falls back
    to 0.0 if nothing is found — this is intentionally conservative rather
    than guessing from date ranges, which is a common source of silent
    extraction errors.
    """
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+of\s+experience",
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+experience",
        r"experience\s*:\s*(\d+(?:\.\d+)?)\+?\s*years?",
    ]
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return float(match.group(1))
    return 0.0


def extract_education_level(text: str) -> str:
    """Return the highest education level keyword found, or 'unknown'."""
    text_lower = text.lower()
    for level, keywords in EDUCATION_LEVELS:
        for kw in keywords:
            if kw in text_lower:
                return level
    return "unknown"


def build_skill_vocab_from_jd(jd_text: str, extra_vocab=None) -> list:
    """Combine the default vocab with any DEFAULT_SKILL_VOCAB terms that
    literally appear in the JD, so scoring stays focused on what this role
    actually asks for rather than an unrelated generic list.
    """
    vocab = extra_vocab or DEFAULT_SKILL_VOCAB
    jd_lower = jd_text.lower()
    return [s for s in vocab if s.lower() in jd_lower] or vocab
