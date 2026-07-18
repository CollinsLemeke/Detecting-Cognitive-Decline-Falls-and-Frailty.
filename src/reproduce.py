
"""
Reproduce the headline results without opening a notebook.

Run from the repository root:

    python src/reproduce.py

Everything is seeded, so the numbers printed here should match the paper
and the notebook exactly.
"""

from __future__ import annotations

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# The six predictors that survive the redundancy check (see Section 3 of the README).
FEATURES = ["age", "bmi", "tug", "step_speed", "clearance", "stridetime_std"]

# Everything we considered before removing collinear duplicates.
CANDIDATE_FEATURES = [
    "age", "bmi", "step_speed", "stride_len", "cadence",
    "clearance", "clearance_std", "swing", "stridetime_std", "tug",
]


# --------------------------------------------------------------------------- #
# Data loading and cleaning
# --------------------------------------------------------------------------- #

def find_data_file() -> str:
    """Locate the register file from the repo root, a subfolder, or Colab."""
    candidates = [
        "data/Database_register.xlsx",
        "../data/Database_register.xlsx",
        "/content/Database_register.xlsx",
        "Database_register.xlsx",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find Database_register.xlsx. Run this from the repository root."
    )


def load_register(path: str) -> tuple[pd.DataFrame, list]:
    """
    Read the GSTRIDE register.

    The file carries three header rows (category, variable name, description),
    so the real column names are on row 1 and the data starts on row 3.
    """
    if path.lower().endswith(".csv"):
        full = pd.read_csv(path, sep=";", encoding="cp1252", header=None)
    else:
        full = pd.read_excel(path, header=None)

    colnames = full.iloc[1].tolist()
    data = full.iloc[3:].copy()
    data.columns = colnames
    return data.dropna(how="all").reset_index(drop=True), colnames


def build_matrix(df: pd.DataFrame, cols: list, feature_list: list) -> pd.DataFrame:
    """Assemble a predictor matrix from the raw register."""

    def find(key: str):
        hits = [c for c in cols if isinstance(c, str) and key.lower() in c.lower()]
        return hits[0] if hits else None

    def num(key: str) -> pd.Series:
        return pd.to_numeric(df[find(key)], errors="coerce")

    def age_midpoint(value) -> float:
        digits = re.findall(r"\d+", str(value))
        return float(np.mean([int(d) for d in digits])) if digits else np.nan

    source = {
        "age": df[find("Age (range")].map(age_midpoint),
        "bmi": num("Body-mass"),
        "tug": num("TUG test"),
        "step_speed": num("Step Speed - Avg"),
        "stride_len": num("Stride Length - Avg"),
        "cadence": num("Cadence - Avg"),
        "clearance": num("Clearance - Avg"),
        "clearance_std": num("Clearance - STD"),
        "swing": num("Swing - Avg"),
        "stridetime_std": num("Stride time - STD"),
    }
    return pd.DataFrame({name: source[name] for name in feature_list})


def build_targets(df: pd.DataFrame, cols: list) -> dict[str, pd.Series]:
    """
    Build the three binary outcomes.

    Note the .str.strip() on the fall label: eight rows are recorded as "NO "
    with a trailing space. Without stripping, those eight people fail to map
    and silently drop out of the analysis.
    """

    def find(key: str):
        return [c for c in cols if isinstance(c, str) and key.lower() in c.lower()][0]

    gds = pd.to_numeric(df[find("Global Deterioration")], errors="coerce")
    fried = pd.to_numeric(df[find("Frailty assessment")], errors="coerce")
    falls = (
        df[find("Falls during the last year")]
        .astype(str).str.strip().str.upper()
        .map({"YES": 1, "NO": 0})
    )

    return {
        "Dementia": (gds >= 3).astype(int),
        "Falls": falls.astype(int),
        "Frailty": (fried >= 2).astype(int),
    }


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def make_models() -> dict[str, Pipeline]:
    """Three algorithm families, each behind the same preprocessing pipeline."""
    base = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    return {
        "Logistic Regression": Pipeline(base + [
            ("m", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))
        ]),
        "Random Forest": Pipeline(base + [
            ("m", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline(base + [
            ("m", HistGradientBoostingClassifier(
                max_leaf_nodes=8, learning_rate=0.06, max_iter=150,
                l2_regularization=1.0, random_state=RANDOM_STATE))
        ]),
    }


def header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    path = find_data_file()
    df, cols = load_register(path)
    targets = build_targets(df, cols)

    header("COHORT")
    print(f"Data file : {path}")
    print(f"Sample    : {len(df)} participants")
    for name, y in targets.items():
        print(f"  {name:<10} {int(y.sum()):>3} of {len(y)}  ({100 * y.mean():.0f}% positive)")

    # ---- Redundancy check: why we dropped four gait measures ---------------- #
    header("FEATURE REDUNDANCY (VIF)")
    print("A VIF above 10 means a feature is almost entirely explained by the")
    print("others, which makes its coefficient unstable and its story unreliable.\n")

    candidates = build_matrix(df, cols, CANDIDATE_FEATURES)
    scaled = StandardScaler().fit_transform(candidates.fillna(candidates.median()))
    before = sorted(
        ((c, variance_inflation_factor(scaled, i)) for i, c in enumerate(CANDIDATE_FEATURES)),
        key=lambda t: -t[1],
    )
    print("  Before cleaning:")
    for name, value in before:
        flag = "  <-- severe" if value > 10 else ("  <-- high" if value > 5 else "")
        print(f"    {name:<16} {value:7.1f}{flag}")

    X = build_matrix(df, cols, FEATURES)
    scaled_clean = StandardScaler().fit_transform(X.fillna(X.median()))
    print("\n  After cleaning (kept 6 features):")
    for i, name in enumerate(FEATURES):
        print(f"    {name:<16} {variance_inflation_factor(scaled_clean, i):7.1f}")

    # ---- Model comparison --------------------------------------------------- #
    header("MODEL COMPARISON  (5-fold CV x 10 repeats)")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    winners = {}

    for task, y in targets.items():
        print(f"\n  {task}  ({int(y.sum())}/{len(y)} positive)")
        print(f"    {'algorithm':<22}{'ROC-AUC':<18}{'recall':<8}")
        rows = []
        for name, model in make_models().items():
            scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc", "recall"])
            rows.append((name, scores["test_roc_auc"].mean(),
                         scores["test_roc_auc"].std(), scores["test_recall"].mean()))
        rows.sort(key=lambda r: -r[1])
        for name, auc_mean, auc_sd, recall in rows:
            mark = "  <-- best" if name == rows[0][0] else ""
            print(f"    {name:<22}{auc_mean:.3f} +/- {auc_sd:.3f}   {recall:.3f}{mark}")
        winners[task] = rows[0]

    # ---- Screening performance on the primary task -------------------------- #
    header("SCREENING PERFORMANCE  (dementia, out-of-fold)")
    y_dem = targets["Dementia"]
    proba = cross_val_predict(
        make_models()["Logistic Regression"], X, y_dem,
        cv=StratifiedKFold(5, shuffle=True, random_state=1),
        method="predict_proba",
    )[:, 1]
    cm = confusion_matrix(y_dem, (proba >= 0.5).astype(int))
    caught, missed, false_alarms = cm[1, 1], cm[1, 0], cm[0, 1]
    print(f"  Caught       : {caught}")
    print(f"  Missed       : {missed}")
    print(f"  False alarms : {false_alarms}")
    print(f"  Catch rate   : {100 * caught / (caught + missed):.0f}%")

    # ---- What the model learned --------------------------------------------- #
    header("ODDS RATIOS  (dementia, per 1 SD)")
    print("  Above 1 raises the odds of cognitive decline, below 1 lowers them.\n")
    imputed = X.fillna(X.median())
    standardised = pd.DataFrame(StandardScaler().fit_transform(imputed), columns=FEATURES)
    fitted = sm.Logit(y_dem, sm.add_constant(standardised)).fit(disp=0)

    ratios = sorted(
        ((f, float(np.exp(fitted.params[f])), float(fitted.pvalues[f])) for f in FEATURES),
        key=lambda t: -t[1],
    )
    for name, odds, p_value in ratios:
        mark = "  <-- statistically reliable" if p_value < 0.05 else ""
        print(f"    {name:<16} OR={odds:5.2f}   p={p_value:.3f}{mark}")

    # ---- Are the three tasks the same signal? ------------------------------- #
    header("TASK SEPARABILITY")
    print("  If cognitive decline were simply frailty in disguise, its risk scores")
    print("  would track frailty as tightly as falls do.\n")

    scores = {}
    for task, y in targets.items():
        scores[task] = cross_val_predict(
            make_models()["Logistic Regression"], X, y,
            cv=StratifiedKFold(5, shuffle=True, random_state=1),
            method="predict_proba",
        )[:, 1]

    names = list(targets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r, _ = pearsonr(scores[names[i]], scores[names[j]])
            print(f"    {names[i]:<10} vs {names[j]:<10} r = {r:+.2f}")

    header("DONE")
    print("  These figures should match the notebook and the README exactly.")
    print("  If they do not, please open an issue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
