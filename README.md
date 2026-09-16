# Resume Screening Agent

Ranks a folder of resumes (PDF/DOCX/TXT) against a job description and outputs
a scored, explained shortlist as JSON and CSV.

Built for the Rooman AI Challenge — Junior AI Research Associate selection round.

## What it does

Given a job description and a folder of resumes, the agent:

1. Parses each resume to plain text (PDF, DOCX, or TXT).
2. Extracts structured signals per candidate: matched/missing skills, stated
   years of experience, and education level.
3. Computes a TF-IDF cosine-similarity score between each resume and the JD.
4. Blends similarity + skill overlap + experience + education into one
   weighted `total_score` (0-100) per candidate.
5. Generates a one-line, fact-grounded explanation for each score (via an
   LLM if you provide an API key, or a template if you don't).
6. Outputs a ranked list to `output/ranked_candidates.json` and `.csv`.

## Setup

```bash
git clone <this-repo-url>
cd resume-screening-agent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Optional — for LLM-written reasoning instead of the template fallback:**

```bash
export ANTHROPIC_API_KEY=your_key_here     # Windows: set ANTHROPIC_API_KEY=your_key_here
```

The agent works fully without this key — it just uses a template-based
explanation instead of an LLM-written one. Nothing else in the pipeline
depends on the LLM; scoring and ranking are deterministic either way.

## How to run

```bash
python3 main.py \
  --jd sample_data/job_description.txt \
  --resumes sample_data/resumes \
  --out output \
  --min-years 3 \
  --min-education bachelor
```

To force the no-API-key template mode even if a key is set:

```bash
python3 main.py --jd sample_data/job_description.txt --resumes sample_data/resumes --no-llm
```

### Arguments

| Flag | Required | Description |
|---|---|---|
| `--jd` | yes | Path to the job description file (`.txt`/`.pdf`/`.docx`) |
| `--resumes` | yes | Path to a folder of resume files |
| `--out` | no | Output directory (default: `output/`) |
| `--min-years` | no | Minimum years of experience the JD requires (default: 0) |
| `--min-education` | no | `unknown` / `bachelor` / `master` / `phd` (default: `unknown`) |
| `--no-llm` | no | Skip LLM reasoning even if `ANTHROPIC_API_KEY` is set |

### Output

`output/ranked_candidates.json` — full structured record per candidate:

```json
{
  "filename": "resume_01_priya_sharma.txt",
  "rank": 1,
  "total_score": 61.8,
  "similarity_score": 37.39,
  "skill_score": 76.9,
  "matched_skills": ["python", "django", "postgresql", "docker", "aws", "..."],
  "missing_skills": ["kubernetes", "redis", "sql"],
  "years_experience": 5.0,
  "experience_score": 100.0,
  "education_level": "bachelor",
  "education_score": 100.0,
  "reasoning": "Matches on agile, aws, ci/cd, django, docker, ...; with 5.0 years of stated experience; but is missing kubernetes, redis, sql."
}
```

`output/ranked_candidates.csv` — the same data flattened into one row per
candidate, sorted by rank, for quick review in a spreadsheet.

## Sample data included

- `sample_data/job_description.txt` — a Backend Python Developer JD
- `sample_data/resumes/` — 10 synthetic resumes deliberately spanning strong
  matches, partial matches, an unrelated profile (marketing), and a fresh
  graduate, so the ranking is actually testable rather than trivial.

Running the command above against this sample data reproduces the ranked
output already committed in `output/`.

## Project structure

```
resume-screening-agent/
├── main.py                  # CLI entrypoint — wires everything together
├── src/
│   ├── parser.py             # PDF/DOCX/TXT → plain text
│   ├── extractor.py          # skills / years / education extraction
│   ├── scorer.py              # TF-IDF similarity + weighted scoring
│   └── reasoning.py           # LLM or template explanation per candidate
├── sample_data/
│   ├── job_description.txt
│   └── resumes/               # 10 sample resumes
├── output/                    # generated ranked_candidates.json / .csv
└── requirements.txt
```

## Design decisions and scoring method

**Similarity method: TF-IDF + cosine similarity, not embeddings.** Chosen for
speed, zero external dependencies (no model download, no API call needed to
score), and — most importantly — explainability: every similarity number can
be traced back to actual shared terms between the JD and resume. See
[TRADEOFFS.md](TRADEOFFS.md) for the full reasoning and what an
embeddings-based version would change.

**Score = weighted blend, not a single similarity number.**

| Component | Weight | Why |
|---|---|---|
| TF-IDF similarity | 50% | Captures overall topical overlap |
| Skill keyword overlap | 30% | Directly checkable against the JD's explicit asks |
| Experience match | 15% | Years stated vs. years required |
| Education match | 5% | Lowest weight — least predictive of actual job fit |

The weights are a judgment call, not a derived optimum — see TRADEOFFS.md.

**Reasoning is grounded, not generated freely.** The LLM (when used) is only
ever shown the already-computed facts (matched skills, years, education) and
asked to phrase them, not asked to score the candidate itself. This avoids
the LLM inventing qualifications or drifting from the numeric score.

## Known limitations

- Keyword-based skill extraction misses synonyms/paraphrases (e.g. "led a
  team of engineers" won't match a "management experience" JD requirement).
- Years-of-experience extraction relies on the resume explicitly stating
  "X years of experience" — it does not sum date ranges from a work history
  table, which is a common resume format we don't yet handle.
- Skill vocabulary is a fixed list scoped to software/backend roles; a
  different role family would need a different vocabulary.

Full tradeoff reasoning and what would change with more time is in
[TRADEOFFS.md](TRADEOFFS.md).
