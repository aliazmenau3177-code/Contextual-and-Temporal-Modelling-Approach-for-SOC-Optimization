# Learning the Incident, not the Reporter

**Contextual and temporal modelling for false-positive reduction in Security Operations Centres — evidence from the Microsoft GUIDE benchmark.**

This repository contains the analysis code for our study on reducing false positives in Security Operations Centre (SOC) alert triage. It accompanies the paper of the same name and reproduces every number, table and figure reported there.

---

## What this work is about

SOCs drown in alerts, most of them false positives. Machine learning is the usual proposed remedy, and published models routinely report high accuracy on public benchmarks. But a question is rarely asked: **what are these models actually learning?**

Security data is full of identifiers — which organisation an alert came from (`OrgId`), which detector fired it (`DetectorId`). These carry strong statistical signal, because some detectors and some tenants are simply noisier than others. A model can score very well by learning "detector 47 usually cries wolf" without understanding a single thing about the incident in front of it.

This project sets out to separate genuine incident understanding from that kind of **reporter memorisation**, and asks whether real contextual and temporal signal can take its place. The short version of what we found:

- A conventional strong classifier scores **0.9713 AUC** on this benchmark — but strip out the identifier columns and it collapses to **0.8263**. Roughly half of its apparent skill was knowing *who reported the alert*, not what the alert was.
- Our model — a gradient-boosted classifier fed by an **entity-correlation graph stream** and a **strictly causal temporal stream** — scores **0.9879 AUC with the identifiers removed entirely**, and cuts the false-positive rate at a fixed 90% threat recall from **0.4494 to 0.0291 (a 93.5% reduction)**. When behavioural context is modelled directly, detector identity becomes almost worthless: the model barely notices when it is taken away.
- We also report, honestly, what **did not** work. Unsupervised anomaly detection (Isolation Forest / LOF), a learned "adaptive fusion gate", and ensembling with CatBoost were all tested under the same protocol and added nothing. The widespread intuition that stacking classifiers with anomaly detectors meaningfully reduces SOC false positives is not supported by our experiments — and we show why.

The larger aim is to lay one foundation stone toward a contextually aware, semi-autonomous SOC — while being careful to claim only what the evidence supports.

---

## The dataset

We use **Microsoft GUIDE**, the largest public SOC incident-triage dataset, released for research in 2024–2025. It ships as an official train/test split totalling ~13.6 million evidence rows.

Download it from Kaggle:
**https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction**

### An important note on how the data is grouped

GUIDE's rows are **evidence**, not incidents — many rows describe a single incident and share its label. A subtlety we document in the paper (and that this code handles): the `IncidentId` field is **scoped within an organisation, not globally unique**. Grouping on `IncidentId` alone merges unrelated incidents from different tenants, which corrupts ~12.8% of incident labels and creates over 109,000 phantom train/test collisions. The code groups on the composite key **(`OrgId`, `IncidentId`)**, which resolves both problems exactly. If you build your own pipeline on GUIDE, this is worth knowing before you start.

### Working with a smaller subset

The full dataset is large, and running the whole thing needs real memory. If you just want to try the pipeline, we provide a balanced subset:

**https://drive.google.com/file/d/1NTa6pRXkCLhJL3oW_tn-Mq2u2yNfHZNo/view?usp=sharing**

Download the zip, extract the CSV, and point the script at it (see below). The numbers you get on a subset will differ from the paper's headline figures, which are computed on the full official test set — but the pipeline, the ablations and the figures all run identically.

---

## What the code does

The single script, `script4_attribution.py`, runs the complete experiment end to end. It is deterministic given the fixed random seed, and it prints its own results and writes its own figures — nothing else to run.

In order, it:

1. **Loads** the official GUIDE train and test partitions.
2. **Diagnoses the grouping key** — measures label consistency and train/test overlap for `IncidentId`, `(OrgId, IncidentId)` and `(OrgId, IncidentId, AlertId)`, and reports which is clean.
3. **Builds incident-level features** by aggregating each incident's evidence rows (evidence counts, entity diversity, MITRE technique counts, category, timing).
4. **Builds the graph stream** — an entity–incident graph over shared devices, IPs, accounts, files and URLs, giving each incident its neighbourhood structure and neighbour threat rate. The graph is built from **training incidents only**, with leave-one-out neighbour statistics, so no incident can see its own label.
5. **Builds the sequence stream** — strictly causal per-organisation alert history (time since last incident, rolling counts over 1h/24h/7d, burst rate, prior threat rate). Every feature uses only incidents that occurred *earlier* in time; test labels are never touched.
6. **Runs the full ablation ladder twice** — once *with* provenance (`OrgId`, `DetectorId`) and once *content-only* (provenance removed). This dual run is the heart of the study: the content-only column is where the real claim lives.
7. **Decomposes the sequence stream** into structural-timing-only, reputation-only, and full variants, to show exactly which signal is doing the work.
8. **Evaluates** at a matched 90% threat recall (so false positives can't be reduced simply by catching fewer threats), with 5-fold cross-validation, McNemar significance testing (with the direction of the result stated explicitly), and a provenance-dependence measurement.
9. **Prints an attribution verdict** and **writes seven figures** (attribution, FPR comparison, feature importances, content-only ROC, per-configuration metrics, analyst workload, and the proposed model's confusion matrix).

### The proposed model

A single **LightGBM** classifier over the concatenation of incident, graph and sequence features. Not four classifiers; not an anomaly ensemble; not a fusion gate. The contribution is the **feature streams and the evaluation protocol**, not the learner — the paper is explicit that gradient boosting, graph features and temporal features are all standard, and that what matters is what they reveal when tested strictly.

---

## Requirements

**Environment**

- Python 3.10 or later
- A machine with at least 16 GB RAM (more comfortable at 32 GB for the full dataset). The full run is memory-heavy because of the graph index over millions of rows; the subset runs on modest hardware.
- We developed and ran this on Ubuntu with PyCharm, but any Python IDE or a plain terminal works.

**Python packages**

```
numpy
pandas
scikit-learn
scipy
matplotlib
lightgbm
catboost
```

Install them with:

```bash
pip install numpy pandas scikit-learn scipy matplotlib lightgbm catboost
```

`lightgbm` and `catboost` are important: without them the script silently falls back to a scikit-learn substitute, and the models will not match the paper's named classifiers. Install both.

---

## How to run

1. Download the dataset (full or subset, above).
2. Open `script4_attribution.py` and set the two paths near the top:

   ```python
   TRAIN_PATH = Path("GUIDE_Train.csv")   # <-- your training CSV
   TEST_PATH  = Path("GUIDE_Test.csv")    # <-- your test CSV
   ```

3. Run it:

   ```bash
   python script4_attribution.py
   ```

The script prints the grouping-key diagnostic, the full ablation table (with and without provenance), the attribution verdict, the McNemar results and the top features, then writes seven figures and a `summary.json` to an output folder. On the full dataset the run takes on the order of an hour or two depending on your hardware; the subset is much faster.

---

## Reproducibility

Everything is deterministic given the published random seed (`42`). The grouping-key diagnostic, feature construction, ablation ladder, statistical tests and figure generation are all contained in the one script, so a single run reproduces the paper's results. If you use a subset rather than the full official split, expect the *shape* of the findings to hold (provenance dependence in the baseline, its near-elimination by the streams, the failed components) while the exact decimals will differ.

---

## A note on honesty

This project reports negative results deliberately. Isolation Forest and LOF, the fusion gate, and CatBoost ensembling are all in the ablation not because they worked, but because they didn't — and showing that clearly is part of the contribution. If you extend this code, we'd encourage keeping that discipline: run the provenance-ablated column, and report what your model looks like *without* the identifiers. A high score with them is easy; a high score without them is the one that means something.

---

## Citation

If you use this code or build on this work, please cite the accompanying paper:

```
[Author list]. Learning the Incident, not the Reporter: Contextual and Temporal
Modelling Approach Combined with Gradient-Boosted Classification for
False-Positive Reduction in Security Operations Centres — Evidence from the
Microsoft GUIDE Benchmark. [Journal], [Year].
```

And the GUIDE dataset it relies on:

```
Freitas, S., et al. AI-Driven Guided Response for Security Operation Centers
with Microsoft Copilot for Security. arXiv:2407.09017 (2024).
```

---

## Contact

[Your name / email / institution — to complete.]
