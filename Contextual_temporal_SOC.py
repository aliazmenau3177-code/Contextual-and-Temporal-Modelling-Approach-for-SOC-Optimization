#!/usr/bin/env python3
"""
Script 4 - FINAL RUN: mechanism attribution.

WHY THIS RUN EXISTS
-------------------
Script 3 established the headline result: with provenance (OrgId/DetectorId)
REMOVED, graph+sequence features lift AUC .8263 -> .9879 and cut FPR at matched
recall from .4494 to .0291 (a 93.5% relative FP reduction). Claim 2 (contextual
+ temporal awareness) is supported.

ONE VULNERABILITY REMAINS. The sequence stream contains `s_prior_threat_rate` -
the organisation's historical true-positive rate over its PAST incidents. It is
computed causally (shifted; an incident never sees its own or any future label),
so it is NOT leakage. But it is arguably an ORG REPUTATION signal - i.e.
provenance wearing temporal clothing. A reviewer will ask: "you claim to have
removed provenance, but did you smuggle OrgId back in through the history?"

This run answers that question directly by decomposing the sequence stream:

    seq_full     all 7 sequence features                    (what Script 3 ran)
    seq_norep    s_prior_threat_rate REMOVED                <- purity test
                 (inter-arrival gaps, rolling 1h/24h/7d counts, burst rate,
                  prior-incident count = STRUCTURAL TIMING ONLY)
    seq_reponly  ONLY s_prior_threat_rate                   <- attribution test

INTERPRETATION (decided in advance, so we cannot rationalise after the fact):
  * If content-only (+graph+seq_norep) stays ~.98  -> the result is driven by
    structural context and timing. The claim is bulletproof. Report it.
  * If it collapses toward ~.90 and seq_reponly alone is strong -> much of the
    "temporal" lift is org reputation. The result still stands, but we must
    describe the mechanism honestly as including a reputational component.
  Either way we report what we find.

ALSO IN THIS RUN
  * Per-feature gain importances for the graph + sequence features, so the paper
    can say WHICH context signals matter rather than hand-waving.
  * McNemar for the PROPOSED model (+graph+seq) vs the strong baseline, in both
    provenance settings, with the direction ("FAVOURS") stated explicitly.

REMOVED (reported as NEGATIVE RESULTS from earlier runs, not re-run):
  * IsolationForest / LOF anomaly fusion  - contributed nothing (Script 2:
    +gate .9741 vs +gate_noanom .9742).
  * The context-adaptive fusion gate      - Script 3: content-only +gate_cal
    .9874 < +graph+seq .9879, and it LOST McNemar to the baseline
    (3,678 vs 20,621). It does not earn its place.
  * CatBoost averaging is KEPT as `ensemble` only to re-verify that naive
    averaging does not beat a single well-fed LightGBM.

PROPOSED MODEL = single LightGBM over [incident features + graph stream +
                 sequence stream].

NOTHING here spins results. It measures and reports.
"""

import json, re, sys, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, roc_curve, precision_recall_curve,
                             confusion_matrix)

warnings.filterwarnings("ignore")

# ============================== CONFIG =======================================
TRAIN_PATH = Path("GUIDE_Train.csv")   # <-- SET ME
TEST_PATH  = Path("GUIDE_Test.csv")    # <-- SET ME
OUT_DIR    = Path("script4_out")
DELIMITER  = ","

GROUP_KEYS = ["OrgId", "IncidentId"]      # validated: 0% mixed, 0 overlap
LABEL_COL  = "IncidentGrade"
TIME_COL   = "Timestamp"
ORG_COL    = "OrgId"
THREAT_GRADES = {"TruePositive"}

SEMANTIC_CATS = ["Category", "EntityType", "EvidenceRole"]
DISTINCT_COUNT_COLS = ["DeviceId", "IpAddress", "AccountSid", "Url",
                       "Sha256", "FileName", "DetectorId"]
ENTITY_COLS = ["DeviceId", "IpAddress", "AccountSid", "Sha256", "Url", "FileName"]
TECH_COL   = "MitreTechniques"
IDENTIFIER_COLS = ["OrgId", "DetectorId"]     # provenance

# the feature under investigation
REPUTATION_FEATURE = "s_prior_threat_rate"

MATCHED_RECALL = 0.90
CV_FOLDS       = 5
RANDOM_SEED    = 42
MAX_ENTITY_DEGREE = 10_000
# =============================================================================

rng = np.random.default_rng(RANDOM_SEED)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def make_gbm(kind="lgbm", seed=RANDOM_SEED):
    if kind == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                  num_leaves=64, subsample=0.8,
                                  colsample_bytree=0.8, random_state=seed,
                                  n_jobs=-1, verbose=-1)
        except Exception:
            pass
    if kind == "catboost":
        try:
            from catboost import CatBoostClassifier
            return CatBoostClassifier(iterations=300, learning_rate=0.05,
                                      depth=8, random_seed=seed, verbose=0)
        except Exception:
            pass
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          random_state=seed)


def backend_report():
    have = {}
    for m in ("lightgbm", "catboost"):
        try:
            __import__(m); have[m] = True
        except Exception:
            have[m] = False
    if not all(have.values()):
        return ("WARNING: " + ", ".join(k for k, v in have.items() if not v) +
                " missing -> sklearn fallback. pip install lightgbm catboost.")
    return "Using LightGBM + CatBoost."


TECH_SPLIT = re.compile(r"[;,\s]+")
def n_techniques(v):
    if pd.isna(v):
        return 0
    s = str(v)
    for ch in "[]'\"":
        s = s.replace(ch, " ")
    return len([t for t in TECH_SPLIT.split(s.strip()) if t])


def make_gkey(df, keys):
    s = df[keys[0]].astype(str)
    for k in keys[1:]:
        s = s + "||" + df[k].astype(str)
    return s


def needed_columns():
    cols = [LABEL_COL, TIME_COL, TECH_COL] + GROUP_KEYS + SEMANTIC_CATS + \
        DISTINCT_COUNT_COLS + ENTITY_COLS + IDENTIFIER_COLS
    return list(dict.fromkeys(cols))


def load(path):
    want = needed_columns()
    df = pd.read_csv(path, sep=DELIMITER, dtype=str, low_memory=False,
                     usecols=lambda c: c.strip() in want)
    df.columns = [c.strip() for c in df.columns]
    df = df[df[LABEL_COL].notna()].copy()
    df["_y"] = df[LABEL_COL].isin(THREAT_GRADES).astype(int)
    df["_gkey"] = make_gkey(df, GROUP_KEYS)
    return df


# --------------------- base incident-level features ---------------------------
def build_base_features(df):
    df = df.copy()
    df["_ntech"] = df[TECH_COL].map(n_techniques) if TECH_COL in df else 0
    df["_ts"] = pd.to_datetime(df[TIME_COL], errors="coerce")

    g = df.groupby("_gkey", sort=False)
    F = pd.DataFrame(index=g.size().index)
    F["_y"] = g["_y"].max()
    F["_org"] = g[ORG_COL].first()
    F["_t"] = g["_ts"].min()

    F["n_evidence"] = g.size()
    for c in DISTINCT_COUNT_COLS:
        if c in df:
            F[f"nd_{c}"] = g[c].nunique()
    F["ntech_max"] = g["_ntech"].max()
    F["ntech_mean"] = g["_ntech"].mean()
    if TECH_COL in df:
        F["nd_techpattern"] = g[TECH_COL].nunique()
    for c in SEMANTIC_CATS:
        if c in df:
            F[f"cat_{c}"] = g[c].agg(
                lambda s: s.mode().iat[0] if len(s.mode()) else "NA")
    tmin, tmax = g["_ts"].min(), g["_ts"].max()
    F["timespan_s"] = (tmax - tmin).dt.total_seconds().fillna(0)
    F["hour"] = tmin.dt.hour.fillna(-1)
    F["dow"] = tmin.dt.dayofweek.fillna(-1)
    F["n_days"] = g["_ts"].agg(lambda s: s.dt.normalize().nunique())
    for c in IDENTIFIER_COLS:
        if c in df:
            F[f"id_{c}"] = g[c].agg(
                lambda s: s.mode().iat[0] if len(s.mode()) else "NA")
    return F


# =============================== GRAPH STREAM ================================
def build_entity_index(df_train):
    """entity -> set of TRAIN incident gkeys. TRAIN ONLY (test only looks up).
    Building over train+test would leak test structure/labels."""
    ent2inc = defaultdict(set)
    for col in ENTITY_COLS:
        if col not in df_train:
            continue
        sub = df_train[[col, "_gkey"]].dropna()
        for val, gk in zip(sub[col].values, sub["_gkey"].values):
            if val in ("", "nan"):
                continue
            ent2inc[f"{col}::{val}"].add(gk)
    return {e: s for e, s in ent2inc.items() if 1 < len(s) <= MAX_ENTITY_DEGREE}


def graph_features(df, ent2inc, train_labels, is_train):
    inc_ents = defaultdict(set)
    for col in ENTITY_COLS:
        if col not in df:
            continue
        sub = df[[col, "_gkey"]].dropna()
        for val, gk in zip(sub[col].values, sub["_gkey"].values):
            if val in ("", "nan"):
                continue
            key = f"{col}::{val}"
            if key in ent2inc:
                inc_ents[gk].add(key)

    rows = {}
    for gk, ents in inc_ents.items():
        degs = [len(ent2inc[e]) for e in ents]
        neigh = set()
        for e in ents:
            neigh |= ent2inc[e]
        if is_train:
            neigh.discard(gk)                     # leave-one-out: no self-label
        ys = [train_labels[n] for n in neigh if n in train_labels]
        rows[gk] = {
            "g_n_shared_entities": len(ents),
            "g_deg_max": max(degs),
            "g_deg_mean": float(np.mean(degs)),
            "g_deg_sum": int(np.sum(degs)),
            "g_n_neighbours": len(neigh),
            "g_neigh_threat_rate": float(np.mean(ys)) if ys else -1.0,
            "g_neigh_labelled": len(ys),
        }
    cols = ["g_n_shared_entities", "g_deg_max", "g_deg_mean", "g_deg_sum",
            "g_n_neighbours", "g_neigh_threat_rate", "g_neigh_labelled"]
    G = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=cols)
    return G, cols


# ============================= SEQUENCE STREAM ===============================
SEQ_COLS = ["s_dt_prev", "s_prior_count", "s_prior_threat_rate",
            "s_cnt_1h", "s_cnt_24h", "s_cnt_7d", "s_burst"]


def _seq_core(allF):
    """Shared causal computation. allF must have _org, _t, _y (NaN for test)."""
    g = allF.groupby("_org", sort=False)
    allF["s_dt_prev"] = g["_t"].diff().dt.total_seconds()
    allF["s_prior_count"] = g.cumcount()
    # expanding mean of y, SHIFTED -> excludes the incident's own label
    allF["s_prior_threat_rate"] = (
        g["_y"].apply(lambda s: s.shift().expanding().mean()).values)
    for win, name in ((pd.Timedelta("1h"), "s_cnt_1h"),
                      (pd.Timedelta("24h"), "s_cnt_24h"),
                      (pd.Timedelta("7d"), "s_cnt_7d")):
        vals = []
        for _, sub in allF.groupby("_org", sort=False):
            t = sub["_t"].values
            lo = np.searchsorted(t, t - win.to_timedelta64(), side="left")
            vals.append(pd.Series(np.arange(len(t)) - lo, index=sub.index))
        allF[name] = pd.concat(vals).reindex(allF.index)
    allF["s_burst"] = allF["s_cnt_24h"] / (allF["s_dt_prev"].fillna(3600)/3600 + 1)
    allF["s_dt_prev"] = allF["s_dt_prev"].fillna(-1)
    allF["s_prior_threat_rate"] = allF["s_prior_threat_rate"].fillna(-1)
    return allF


def sequence_features_train(F):
    S = F[["_org", "_t", "_y"]].copy()
    S["_orig"] = np.arange(len(S))
    S = S.sort_values(["_org", "_t"], kind="mergesort")
    S = _seq_core(S)
    out = S.sort_values("_orig")[SEQ_COLS]
    out.index = F.index
    return out


def sequence_features_test(F_test, F_train):
    """Test history comes from TRAIN incidents of the same org. Test labels are
    NEVER used - set to NaN before the expanding mean."""
    tr = F_train[["_org", "_t", "_y"]].copy(); tr["_is_test"] = 0
    te = F_test[["_org", "_t", "_y"]].copy(); te["_is_test"] = 1
    te["_y"] = np.nan
    allF = pd.concat([tr, te])
    allF["_orig"] = np.arange(len(allF))
    allF = allF.sort_values(["_org", "_t"], kind="mergesort")
    allF = _seq_core(allF)
    out = allF.sort_values("_orig")
    out = out[out["_is_test"] == 1][SEQ_COLS]
    out.index = F_test.index
    return out


# ------------------------------ encoding -------------------------------------
class Encoder:
    def __init__(self, use_identifiers=True):
        self.use_identifiers = use_identifiers

    def fit(self, F):
        self.cat_levels = {c: sorted(F[c].dropna().astype(str).unique())
                           for c in F.columns if c.startswith("cat_")}
        self.freq = {}
        if self.use_identifiers:
            for c in [c for c in F.columns if c.startswith("id_")]:
                self.freq[c] = F[c].value_counts().to_dict()
        self.num_cols = [c for c in F.columns
                         if not c.startswith(("cat_", "id_", "_"))]
        return self

    def transform(self, F):
        parts = [F[self.num_cols].astype(float).reset_index(drop=True)]
        for c, levels in self.cat_levels.items():
            oh = pd.get_dummies(pd.Categorical(F[c].astype(str), categories=levels),
                                prefix=c).astype(float).reset_index(drop=True)
            parts.append(oh)
        if self.use_identifiers:
            for c, fmap in self.freq.items():
                parts.append(pd.Series(F[c].map(fmap).fillna(0).values,
                                       name=f"freq_{c}").reset_index(drop=True))
        X = pd.concat(parts, axis=1).fillna(0)
        return X.values.astype(np.float32), list(X.columns)


# ------------------------------- metrics -------------------------------------
def fpr_at_recall(y, p, target):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    if len(ok) == 0:
        return 1.0, 0.0, float(tpr.max())
    i = ok[0]
    return float(fpr[i]), float(thr[i]), float(tpr[i])


def eval_probs(y, p, target=MATCHED_RECALL):
    fpr, thr, rec = fpr_at_recall(y, p, target)
    yhat = (p >= thr).astype(int)
    return {"auc": float(roc_auc_score(y, p)),
            "ap": float(average_precision_score(y, p)),
            "fpr_at_recall": fpr, "thr": float(thr), "recall_at_thr": rec,
            "precision_at_thr": float(precision_score(y, yhat, zero_division=0)),
            "f1_at_thr": float(f1_score(y, yhat)),
            "workload_reduction": float((yhat == 0).mean()),
            "alerts_reviewed": int(yhat.sum()),
            "missed_threats": int(((yhat == 0) & (y == 1)).sum())}


def mcnemar(y, p_prop, p_base, thr_prop, thr_base):
    e_p = (p_prop >= thr_prop).astype(int) != y
    e_b = (p_base >= thr_base).astype(int) != y
    b = int(np.sum(e_p & ~e_b))     # proposed wrong, base right
    c = int(np.sum(~e_p & e_b))     # proposed right, base wrong
    if b + c == 0:
        return {"proposed_only_right": c, "base_only_right": b,
                "stat": 0.0, "p_value": 1.0, "favours": "tie"}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(sstats.chi2.sf(stat, 1))
    favours = ("PROPOSED" if c > b else "BASE" if b > c else "tie")
    return {"proposed_only_right": c, "base_only_right": b,
            "stat": float(stat), "p_value": p, "favours": favours}


# ------------------------------- pipeline ------------------------------------
def run(F_tr, F_te, gcols, seq_variants, use_ids, tag):
    """seq_variants: dict name -> list of sequence columns to include."""
    ytr = F_tr["_y"].values.astype(int)
    yte = F_te["_y"].values.astype(int)
    all_seq = SEQ_COLS
    base_cols = [c for c in F_tr.columns
                 if not c.startswith(("g_", "s_", "_"))]

    def subset(F, extra):
        keep = [c for c in F.columns
                if (c in base_cols or c in extra) and not c.startswith("_")]
        return F[keep + ["_y"]].copy()

    # the ablation ladder for this run
    configs = {
        "base":                     [],
        "+graph":                   gcols,
        "+seq_full":                seq_variants["seq_full"],
        "+seq_norep":               seq_variants["seq_norep"],
        "+seq_reponly":             seq_variants["seq_reponly"],
        "+graph+seq_full":          gcols + seq_variants["seq_full"],
        "+graph+seq_norep":         gcols + seq_variants["seq_norep"],
    }

    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    test_probs, cv_rows, importances = {}, [], {}
    for name, extra in configs.items():
        tr_s, te_s = subset(F_tr, extra), subset(F_te, extra)
        enc = Encoder(use_ids).fit(tr_s)
        Xtr, cols = enc.transform(tr_s)
        Xte, _ = enc.transform(te_s)
        for k, (tri, vai) in enumerate(skf.split(Xtr, ytr)):
            m = make_gbm("lgbm").fit(Xtr[tri], ytr[tri])
            cv_rows.append({"fold": k, "config": name,
                            **eval_probs(ytr[vai], m.predict_proba(Xtr[vai])[:, 1])})
        M = make_gbm("lgbm").fit(Xtr, ytr)
        test_probs[name] = M.predict_proba(Xte)[:, 1]
        if name == "+graph+seq_full":
            vals = None
            if hasattr(M, "feature_importances_"):
                vals = np.asarray(M.feature_importances_, dtype=float)
            else:
                # sklearn HistGradientBoosting has no native importances ->
                # fall back to permutation-free proxy: skip rather than fake it
                print(f"  [{tag}] (no native feature_importances_ on this "
                      f"backend; importance table skipped)")
            if vals is not None and len(vals) == len(cols):
                importances = {c: float(v) for c, v in
                               sorted(zip(cols, vals), key=lambda kv: -kv[1])}
        print(f"  [{tag}] {name:20} test AUC="
              f"{roc_auc_score(yte, test_probs[name]):.4f}", flush=True)

    # ensemble: re-verify the negative result (CatBoost averaging)
    full_tr = subset(F_tr, gcols + all_seq); full_te = subset(F_te, gcols + all_seq)
    enc = Encoder(use_ids).fit(full_tr)
    Xtr, _ = enc.transform(full_tr); Xte, _ = enc.transform(full_te)
    MC = make_gbm("catboost").fit(Xtr, ytr)
    test_probs["ensemble(+catboost)"] = (
        test_probs["+graph+seq_full"] + MC.predict_proba(Xte)[:, 1]) / 2
    print(f"  [{tag}] {'ensemble(+catboost)':20} test AUC="
          f"{roc_auc_score(yte, test_probs['ensemble(+catboost)']):.4f}", flush=True)

    metrics = {n: eval_probs(yte, p) for n, p in test_probs.items()}
    return {"test_probs": test_probs, "metrics": metrics, "yte": yte,
            "cv": pd.DataFrame(cv_rows), "importances": importances}


ORDER = ["base", "+graph", "+seq_reponly", "+seq_norep", "+seq_full",
         "+graph+seq_norep", "+graph+seq_full", "ensemble(+catboost)"]


# ------------------------------- figures -------------------------------------
def figures(r_ids, r_con, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    tm, tc = r_ids["metrics"], r_con["metrics"]
    cfgs = [c for c in ORDER if c in tm]
    x = np.arange(len(cfgs)); w = .38; vir = plt.cm.viridis

    # fig1: THE attribution figure - content-only AUC across sequence variants
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w/2, [tm[c]["auc"] for c in cfgs], w,
           label="with provenance", color=vir(.25))
    ax.bar(x + w/2, [tc[c]["auc"] for c in cfgs], w,
           label="content-only (provenance removed)", color=vir(.7))
    ax.axhline(tc["base"]["auc"], ls="--", lw=.8, color="grey",
               label=f"content-only base ({tc['base']['auc']:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=20, ha="right")
    ax.set_ylim(.75, 1.0); ax.set_ylabel("AUC")
    ax.set_title("Mechanism attribution: is the temporal lift structure or reputation?")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_dir / "fig1_attribution.png", dpi=140); plt.close(fig)

    # fig2: FPR at matched recall, dual
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - w/2, [tm[c]["fpr_at_recall"] for c in cfgs], w,
           label="with provenance", color=vir(.25))
    ax.bar(x + w/2, [tc[c]["fpr_at_recall"] for c in cfgs], w,
           label="content-only", color=vir(.7))
    ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=20, ha="right")
    ax.set_ylabel(f"FPR @ recall={MATCHED_RECALL} (lower better)")
    ax.set_title("False-positive rate at matched recall")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_dir / "fig2_fpr_dual.png", dpi=140); plt.close(fig)

    # fig3: feature importance of the proposed model
    imp = r_ids["importances"]
    if imp:
        top = list(imp.items())[:20][::-1]
        fig, ax = plt.subplots(figsize=(7, 6))
        names = [k for k, _ in top]; vals = [v for _, v in top]
        colors = [vir(.75) if n.startswith("g_") else
                  vir(.45) if n.startswith("s_") else vir(.15) for n in names]
        ax.barh(names, vals, color=colors)
        ax.set_xlabel("LightGBM gain importance")
        ax.set_title("Top features - proposed model (+graph+seq)\n"
                     "green=graph  teal=sequence  dark=base")
        fig.tight_layout(); fig.savefig(out_dir / "fig3_importance.png", dpi=140)
        plt.close(fig)

    # fig4: ROC of proposed vs base (content-only - the honest setting)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, col in (("base", vir(.15)), ("+graph+seq_norep", vir(.5)),
                   ("+graph+seq_full", vir(.8))):
        if c in r_con["test_probs"]:
            fpr, tpr, _ = roc_curve(r_con["yte"], r_con["test_probs"][c])
            ax.plot(fpr, tpr, color=col, label=f"{c} ({tc[c]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.7)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC - CONTENT-ONLY (provenance removed)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_dir / "fig4_roc_content_only.png", dpi=140); plt.close(fig)

    # fig5: four metrics, proposed model, both settings
    mets = [("precision_at_thr", "Precision"), ("recall_at_thr", "Recall"),
            ("f1_at_thr", "F1"), ("fpr_at_recall", "FPR (lower=better)")]
    fig, ax = plt.subplots(figsize=(max(10, 1.4*len(cfgs)), 5))
    ww = .2
    for j, (k, lbl) in enumerate(mets):
        vals = [tm[c][k] for c in cfgs]
        bars = ax.bar(x + (j-1.5)*ww, vals, ww, label=lbl, color=vir(j/4))
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x()+b_.get_width()/2, v+.01, f"{v:.2f}",
                    ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=20, ha="right")
    ax.set_ylim(0, 1.08); ax.set_ylabel("score")
    ax.set_title(f"Metrics by config @ matched recall={MATCHED_RECALL}")
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(.5, -.34))
    fig.tight_layout(); fig.savefig(out_dir / "fig5_metrics.png", dpi=140)
    plt.close(fig)

    # fig6: workload reduction (claim 3)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x, [tm[c]["workload_reduction"]*100 for c in cfgs],
           color=vir(np.linspace(0, .85, len(cfgs))))
    for i, c in enumerate(cfgs):
        ax.text(i, tm[c]["workload_reduction"]*100+.4,
                f"{tm[c]['workload_reduction']*100:.1f}%", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=20, ha="right")
    ax.set_ylabel("% incidents auto-dismissed")
    ax.set_title(f"Analyst workload reduction at {MATCHED_RECALL:.0%} threat recall")
    fig.tight_layout(); fig.savefig(out_dir / "fig6_workload.png", dpi=140)
    plt.close(fig)

    # fig7: confusion matrix of proposed model
    best = "+graph+seq_full"
    thr = tm[best]["thr"]
    cm = confusion_matrix(r_ids["yte"], (r_ids["test_probs"][best] >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred benign", "pred threat"])
    ax.set_yticklabels(["benign", "threat"])
    ax.set_title(f"Proposed model @ recall={MATCHED_RECALL}")
    fig.tight_layout(); fig.savefig(out_dir / "fig7_confusion.png", dpi=140)
    plt.close(fig)


# --------------------------------- main --------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        sys.exit("Set TRAIN_PATH / TEST_PATH.")

    banner("LOAD")
    print(backend_report())
    train, test = load(TRAIN_PATH), load(TEST_PATH)
    print(f"labelled rows  train={len(train):,}  test={len(test):,}")

    banner("BUILD INCIDENT FEATURES")
    F_tr, F_te = build_base_features(train), build_base_features(test)
    print(f"train incidents={len(F_tr):,}  test incidents={len(F_te):,}")
    print(f"threat prevalence  train={F_tr['_y'].mean():.3%}  "
          f"test={F_te['_y'].mean():.3%}")

    banner("GRAPH STREAM (train-only entity index)")
    ent2inc = build_entity_index(train)
    print(f"entities indexed: {len(ent2inc):,}")
    train_labels = F_tr["_y"].to_dict()
    G_tr, gcols = graph_features(train, ent2inc, train_labels, is_train=True)
    G_te, _ = graph_features(test, ent2inc, train_labels, is_train=False)
    G_tr = G_tr.reindex(F_tr.index); G_te = G_te.reindex(F_te.index)
    print(f"incidents with >=1 shared entity:  "
          f"train={G_tr['g_n_shared_entities'].notna().mean():.1%}  "
          f"test={G_te['g_n_shared_entities'].notna().mean():.1%}")
    G_tr = G_tr.fillna({"g_neigh_threat_rate": -1}).fillna(0)
    G_te = G_te.fillna({"g_neigh_threat_rate": -1}).fillna(0)

    banner("SEQUENCE STREAM (causal) + DECOMPOSITION")
    S_tr = sequence_features_train(F_tr)
    S_te = sequence_features_test(F_te, F_tr)
    seq_variants = {
        "seq_full":    SEQ_COLS,
        "seq_norep":   [c for c in SEQ_COLS if c != REPUTATION_FEATURE],
        "seq_reponly": [REPUTATION_FEATURE],
    }
    for k, v in seq_variants.items():
        print(f"  {k:12} -> {v}")

    F_tr = pd.concat([F_tr, G_tr, S_tr], axis=1)
    F_te = pd.concat([F_te, G_te, S_te], axis=1)

    banner("ABLATION - WITH PROVENANCE")
    r_ids = run(F_tr, F_te, gcols, seq_variants, True, "with_ids")

    banner("ABLATION - CONTENT-ONLY  <- THE ATTRIBUTION TEST")
    r_con = run(F_tr, F_te, gcols, seq_variants, False, "content_only")

    tm, tc = r_ids["metrics"], r_con["metrics"]
    cfgs = [c for c in ORDER if c in tm]

    banner("RESULTS")
    print(f"{'config':<22}{'AUC(ids)':>9}{'AUC(cont)':>11}"
          f"{'FPR(ids)':>10}{'FPR(cont)':>11}{'F1':>8}{'workload':>10}")
    for c in cfgs:
        print(f"{c:<22}{tm[c]['auc']:>9.4f}{tc[c]['auc']:>11.4f}"
              f"{tm[c]['fpr_at_recall']:>10.4f}{tc[c]['fpr_at_recall']:>11.4f}"
              f"{tm[c]['f1_at_thr']:>8.4f}{tm[c]['workload_reduction']:>9.1%}")

    b_i, b_c = tm["base"]["fpr_at_recall"], tc["base"]["fpr_at_recall"]
    print(f"\nFP reduction vs strong baseline @ recall={MATCHED_RECALL}:")
    for c in cfgs:
        print(f"  {c:<22} with_ids={(b_i-tm[c]['fpr_at_recall'])/b_i*100:+6.1f}%"
              f"   content_only={(b_c-tc[c]['fpr_at_recall'])/b_c*100:+6.1f}%")

    banner("*** ATTRIBUTION VERDICT ***")
    a_base = tc["base"]["auc"]
    a_full = tc["+graph+seq_full"]["auc"]
    a_norep = tc["+graph+seq_norep"]["auc"]
    a_rep = tc["+seq_reponly"]["auc"]
    print(f"content-only base                      AUC={a_base:.4f}")
    print(f"content-only +seq_reponly (reputation) AUC={a_rep:.4f}")
    print(f"content-only +graph+seq_norep (STRUCT) AUC={a_norep:.4f}")
    print(f"content-only +graph+seq_full           AUC={a_full:.4f}")
    print(f"\n  lift from structure alone (norep - base) = {a_norep-a_base:+.4f}")
    print(f"  extra lift from reputation feature       = {a_full-a_norep:+.4f}")
    retained = (a_norep - a_base) / (a_full - a_base) * 100 if a_full > a_base else 0
    print(f"\n  --> STRUCTURE RETAINS {retained:.1f}% OF THE TOTAL LIFT")
    if a_norep >= 0.97:
        print("  --> VERDICT: the result is driven by structural context and "
              "timing.\n      The temporal/contextual claim is CLEAN. "
              "Report the reputation\n      feature as a minor additional signal.")
    elif retained >= 60:
        print("  --> VERDICT: structure carries the majority of the lift, with a "
              "real\n      reputational component. Describe BOTH mechanisms "
              "honestly.")
    else:
        print("  --> VERDICT: much of the 'temporal' lift is ORG REPUTATION, not\n"
              "      structural timing. The claim must be reworded: the model "
              "benefits\n      substantially from organisational prior rates. "
              "REPORT THIS PLAINLY.")

    banner("McNEMAR - proposed (+graph+seq_full) vs strong baseline")
    for tag, r, m in (("with_ids", r_ids, tm), ("content_only", r_con, tc)):
        mc = mcnemar(r["yte"], r["test_probs"]["+graph+seq_full"],
                     r["test_probs"]["base"],
                     m["+graph+seq_full"]["thr"], m["base"]["thr"])
        print(f"  [{tag:12}] proposed-only-right={mc['proposed_only_right']:,}  "
              f"base-only-right={mc['base_only_right']:,}  "
              f"p={mc['p_value']:.3g}  FAVOURS: {mc['favours']}")
    print("  NOTE: chi2 is symmetric. A small p means the models DIFFER. "
          "Read FAVOURS.")

    banner("TOP FEATURES (proposed model, with provenance)")
    if not r_ids["importances"]:
        print("  (unavailable - LightGBM not installed; install it to get this "
              "table, which the paper needs)")
    for i, (k, v) in enumerate(list(r_ids["importances"].items())[:15]):
        kind = ("GRAPH" if k.startswith("g_") else
                "SEQ" if k.startswith("s_") else
                "PROVENANCE" if k.startswith("freq_id_") else "BASE")
        print(f"  {i+1:2}. {k:28} {v:>10.0f}   [{kind}]")

    banner("FIGURES")
    figures(r_ids, r_con, OUT_DIR)
    print(f"seven figures -> {OUT_DIR}/")

    summary = {
        "n_train_incidents": int(len(F_tr)), "n_test_incidents": int(len(F_te)),
        "matched_recall": MATCHED_RECALL,
        "metrics_with_ids": tm, "metrics_content_only": tc,
        "attribution": {
            "content_only_base_auc": a_base,
            "content_only_reputation_only_auc": a_rep,
            "content_only_structure_auc": a_norep,
            "content_only_full_auc": a_full,
            "structure_lift": a_norep - a_base,
            "reputation_extra_lift": a_full - a_norep,
            "structure_share_of_lift_pct": retained,
        },
        "cv_with_ids": {c: {"auc_mean": float(s["auc"].mean()),
                            "auc_std": float(s["auc"].std())}
                        for c, s in r_ids["cv"].groupby("config")},
        "cv_content_only": {c: {"auc_mean": float(s["auc"].mean()),
                                "auc_std": float(s["auc"].std())}
                            for c, s in r_con["cv"].groupby("config")},
        "top_features": dict(list(r_ids["importances"].items())[:30]),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    r_ids["cv"].to_csv(OUT_DIR / "cv_with_ids.csv", index=False)
    r_con["cv"].to_csv(OUT_DIR / "cv_content_only.csv", index=False)
    print(f"summary.json + cv csvs saved. runtime {summary['runtime_sec']}s")


if __name__ == "__main__":
    main()
