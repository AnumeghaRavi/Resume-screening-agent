# Tradeoff Notes

## Why TF-IDF instead of embeddings

I chose TF-IDF + cosine similarity over embedding-based similarity
(e.g. sentence-transformers or an embeddings API) for three reasons:

1. **Explainability.** TF-IDF similarity is a function of literal shared
   terms — I can show a reviewer exactly why two documents scored the way
   they did. Embeddings capture semantic similarity but are effectively a
   black box for a scoring decision that affects a real candidate.
2. **No external dependency at scoring time.** TF-IDF needs no model
   download and no API call, so the core ranking works completely offline
   and free. The LLM is used only for the reasoning *text*, not the score
   itself — if that call fails or the key is missing, scoring is unaffected.
3. **Time budget.** Given the 24-hour window, TF-IDF is a few lines with
   scikit-learn and is reliable immediately, versus tuning an embeddings
   pipeline (model choice, chunking, similarity thresholds) with less
   certainty of finishing.

**What this costs:** TF-IDF is bag-of-words. It won't catch that "shipped
production ML models" and "deployed machine learning systems" mean roughly
the same thing — they share almost no exact tokens. This is the single
biggest quality gap versus an embeddings-based version, and it's the first
thing I'd change with more time.

## Why a weighted blend instead of one score

A single similarity number is easy to game (a resume that's just a wall of
JD keywords would score artificially high) and hard to explain to a hiring
manager. Splitting into similarity / skills / experience / education lets a
reviewer see *which part* of the match is strong or weak, not just a final
number. The weights (50/30/15/5) reflect a judgment that explicit skill
overlap and general topical fit matter most, and education matters least —
this is a defensible starting point, not a tuned optimum. With real
historical hiring data, these weights should be learned or at least
validated against actual hire/no-hire outcomes rather than set by hand.

## Why grounded LLM reasoning, not LLM scoring

The LLM never sees the raw resume and picks a score — it only sees numbers
and extracted facts that were already computed deterministically, and its
job is limited to writing a readable sentence about them. This was a
deliberate choice to keep the actual ranking decision auditable and
reproducible (same input always gives same score), and to avoid the LLM
silently penalizing or rewarding a candidate for something outside the
stated JD criteria.

## What I'd improve with more time

- **Swap in embeddings for similarity, keep TF-IDF as a fallback.** Best of
  both: catch paraphrased skills, but degrade gracefully to something
  explainable if no embedding model/API is available.
- **Sum experience from date ranges**, not just an explicit "X years"
  phrase — most real resumes list roles with start/end dates rather than
  stating total years directly, so the current extractor would under-score
  many real candidates.
- **Expand the skill vocabulary dynamically** using the LLM to pull skill
  terms out of the JD itself, rather than intersecting with a fixed list —
  the current approach misses any skill the JD asks for that isn't already
  in `DEFAULT_SKILL_VOCAB`.
- **Handle resume tables/columns better** — some PDF resumes use multi-column
  layouts that `pdfplumber`'s default text extraction can scramble; a
  layout-aware extraction pass would catch more real-world resumes.
- **Add unit tests** for the extractor functions (skills/years/education),
  which were the most bug-prone part to get right by hand.

## Known failure cases observed while building

- A candidate with strong experience described only in narrative form (no
  literal skill keywords, e.g. "built the payments platform" instead of
  "Python, Django") scores lower than their real fit, because the skill and
  TF-IDF layers both depend on shared vocabulary.
- Multi-word skills phrased differently than the vocabulary entry (e.g.
  "Postgres" vs. "PostgreSQL") are not matched — the extractor does exact
  substring matching on a fixed skill list, not fuzzy matching.
