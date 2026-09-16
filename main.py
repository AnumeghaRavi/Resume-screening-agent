#!/usr/bin/env python3
"""
Resume Screening Agent
Ranks a folder of resumes against a job description and outputs a scored,
explained shortlist as CSV and JSON.

Usage:
    python main.py --jd sample_data/job_description.txt \\
                    --resumes sample_data/resumes \\
                    --out output \\
                    --min-years 3 \\
                    --min-education bachelor \\
                    --no-llm   # optional: skip LLM reasoning even if a key is set
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.parser import load_resumes_from_dir, extract_text
from src.scorer import score_all
from src.reasoning import generate_reasoning

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Rank resumes against a job description.")
    p.add_argument("--jd", required=True, help="Path to the job description file (.txt/.pdf/.docx)")
    p.add_argument("--resumes", required=True, help="Path to a folder of resume files")
    p.add_argument("--out", default="output", help="Output directory (default: output/)")
    p.add_argument("--min-years", type=float, default=0.0, help="Minimum years of experience the JD requires")
    p.add_argument("--min-education", default="unknown",
                    choices=["unknown", "bachelor", "master", "phd"],
                    help="Minimum education level the JD requires")
    p.add_argument("--no-llm", action="store_true", help="Skip LLM reasoning even if ANTHROPIC_API_KEY is set")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading job description from {args.jd} ...")
    jd_text = extract_text(args.jd)

    print(f"Loading resumes from {args.resumes} ...")
    resumes = load_resumes_from_dir(args.resumes)
    if not resumes:
        print("No resumes found. Check the --resumes path and that files are .pdf/.docx/.txt.")
        sys.exit(1)
    print(f"  found {len(resumes)} resume(s)")

    print("Scoring candidates ...")
    ranked = score_all(
        jd_text, resumes,
        jd_min_years=args.min_years,
        jd_education=args.min_education,
    )

    print("Generating reasoning ..." if not args.no_llm else "Generating reasoning (template mode) ...")
    for candidate in ranked:
        candidate["reasoning"] = generate_reasoning(candidate, jd_text, use_llm=not args.no_llm)

    os.makedirs(args.out, exist_ok=True)

    json_path = os.path.join(args.out, "ranked_candidates.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)

    csv_path = os.path.join(args.out, "ranked_candidates.csv")
    df = pd.DataFrame(ranked)
    column_order = [
        "rank", "filename", "total_score", "similarity_score", "skill_score",
        "experience_score", "education_score", "years_experience",
        "education_level", "matched_skills", "missing_skills", "reasoning",
    ]
    df = df[[c for c in column_order if c in df.columns]]
    df.to_csv(csv_path, index=False)

    print(f"\nDone. Wrote:\n  {json_path}\n  {csv_path}\n")
    print("Top candidates:")
    for c in ranked[:5]:
        print(f"  #{c['rank']:>2}  {c['total_score']:>5.1f}  {c['filename']}")


if __name__ == "__main__":
    main()
