# Pop Culture Clash — Project Proposal (Milestone P2)

This project investigates whether pop-culture references help or hurt captions in the New Yorker Caption Contest. The dataset contains more than two million captions spanning roughly 400 contests. Each caption is scored by contest voters, giving us a rich test bed to evaluate how references to celebrities, memes, brands, and events perform relative to straight-ahead jokes.

Our guiding motivation is that writers often reach for topical punchlines, yet it is unclear when these references actually resonate. We aim to provide data-driven guidance on reference strategy: when are they worth the risk, which topics work with which images, and how has the audience appetite changed over time?

This proposal fulfills Milestone P2 requirements: a detailed plan (below) and initial code artefacts under `analysis/`, including a reproducible notebook (`analysis/pop_culture_clash.ipynb`) and the underlying pipeline (`analysis/pop_culture_analysis.py`).

## Research questions

1. **Do captions that reference pop culture score higher on average than non-referential captions?**
2. **Does recency matter?** Do fresh references outperform nostalgic or retro callbacks?
3. **Are certain reference types more effective, and does effectiveness vary by image theme?**
4. **Do pop-culture captions create more polarization than non-referential captions?**
5. **Has the preference for pop-culture references changed over time?**

## Data sources

- **Caption contest summaries:** We rely on the `caption_contest_data` repository, which aggregates contest-level CSV summaries with per-caption vote statistics (`mean`, `funny`, `not_funny`, `votes`, `precision`, etc.). The local copy under `caption-contest-data-gh-pages/` is treated as the canonical cache to keep the analysis reproducible.
- **Contest metadata:** YAML files in the same repository record start dates for contests. When dates are missing, we interpolate using a linear model over contest IDs to approximate the weekly cadence of releases.
- **Pop-culture lexicon:** We curate a keyword list that tags celebrities, brands, memes, shows, and events with their first year of prominence. The lexicon lives in `analysis/pop_culture_analysis.py` and is reused by the notebook.

## Methodology

Our initial methodology balances interpretability and coverage:

- **Reference detection:** Each caption is normalized (lowercased, ASCII folded) and scanned for keyword matches. A hit marks the caption as a pop-culture reference and records the matched categories (e.g., music, politics). We also track the earliest “first year” among matches to approximate recency.
- **Feature engineering:** We compute caption length, word count, reference category mixes, implied recency buckets, and coarse image themes. Themes are inferred by counting topic keywords in the highest-voted captions per contest (e.g., office, medical, politics).
- **Score interpretation:** We work directly with the contest-provided mean score and complement it with distributional statistics (vote spread, precision). This keeps the pipeline simple and highlights the raw performance gap between reference and non-reference captions, while acknowledging that length and contest effects remain a potential confound.
- **Aggregation:** With these features in hand, we analyze differences in mean scores and distributional statistics across reference vs. non-reference captions, recency buckets, and category-theme combinations.

## Analysis plan by research question

- **RQ1 (Overall performance):** Compute the share of captions containing references and compare their mean scores to the control group. We will report confidence intervals or bootstrap standard errors in later milestones if needed.
- **RQ2 (Recency effects):** Bucket references into `fresh (<3y)`, `current (3-10y)`, `nostalgic (10-20y)`, `retro (>20y)`, and `unknown`. Compare their average scores and inspect distributions to see whether recent references consistently outperform throwbacks.
- **RQ3 (Category-theme fit):** For reference captions, produce a two-way aggregation of `primary_category` and inferred `image_theme`. Rank the combinations by mean score and flag the most promising pairings (e.g., tech references in office scenes, sports references in stadium scenes). We will also measure what fraction of top-performing pairings rely on fresh references.
- **RQ4 (Polarization):** Using vote counts, compute a `vote_spread = (funny - not_funny)/votes` metric alongside the provided `precision`. Compare medians, interquartile ranges, and the share of high-spread captions (>0.5 absolute spread) to see whether references provoke stronger love/hate reactions.
- **RQ5 (Temporal trends):** Track the yearly share of pop references, their average performance relative to non-reference captions, and differences across recency buckets. We will highlight turning points where fresh references pull ahead while retro references stagnate.

## Deliverables for Milestone P2

- **Code:** `analysis/pop_culture_analysis.py` encapsulates data loading, lexicon management, feature engineering, and helper functions (themes, polarization, yearly trends). The file can be run as a script to regenerate CSV summaries and figures in `analysis/`.
- **Notebook:** `analysis/pop_culture_clash.ipynb` walks through the pipeline interactively, answering each research question with tables and plots that mirror the plan above.
- **Artifacts:** Running the pipeline produces summary CSVs (`analysis/summary_metrics.json`, `analysis/polarization_summary.csv`, `analysis/recency_summary.csv`, `analysis/reference_type_theme.csv`, `analysis/yearly_trends.csv`) and a figure (`analysis/figures/yearly_pop_culture_trends.png`) that capture our initial findings.

