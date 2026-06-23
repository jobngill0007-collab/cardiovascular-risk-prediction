"""
Cardiovascular Risk Prediction — Model Training
Supports: CDC BRFSS 2022 (LLCP2022.XPT) or UCI Cleveland (heart.csv)
Output:   heart_model.pkl

Run:  python train_model.py
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))

# ── 18-feature schema (order MUST match index.html form) ─────────────────────
FEATURES = [
    "age",            # 0  numeric 18-80
    "sex",            # 1  0=Female 1=Male
    "high_bp",        # 2  0/1
    "high_chol",      # 3  0/1
    "diabetes",       # 4  0/1
    "bmi_cat",        # 5  1=Under 2=Normal 3=Over 4=Obese
    "smoked_ever",    # 6  0/1
    "exercise",       # 7  0/1 activity past 30 days
    "binge_drink",    # 8  0/1
    "low_frvt",       # 9  0/1  fruit <1/day
    "low_veg",        # 10 0/1  veg <1/day
    "sleep_hrs",      # 11 hours 1-24
    "mental_hlth",    # 12 bad days 0-30
    "stroke_current", # 13 0/1
    "gen_health",     # 14 1=Excellent 5=Poor
    "phys_hlth",      # 15 bad days 0-30
    "family_history", # 16 0/1
    "stress",         # 17 0=Low 1=Moderate 2=High
]

# ── BRFSS 2022 column map ─────────────────────────────────────────────────────
BRFSS_MAP = {
    "age":            "_AGE80",
    "sex":            "SEXVAR",
    "high_bp":        "_RFHYPE6",
    "high_chol":      "_RFCHOL3",
    "diabetes":       "DIABETE4",
    "bmi_cat":        "_BMI5CAT",
    "smoked_ever":    "_SMOKER3",
    "exercise":       "_TOTINDA",
    "binge_drink":    "_RFBING6",
    "low_frvt":       "_FRTLT1A",
    "low_veg":        "_VEGLT1A",
    "sleep_hrs":      "SLEPTIM1",
    "mental_hlth":    "MENTHLTH",
    "stroke_current": "CVDSTRK3",
    "gen_health":     "GENHLTH",
    "phys_hlth":      "PHYSHLTH",
    "family_history": "CVDINFR4",
    "stress":         "_MENT14D",
}
BRFSS_TARGET = "MICHD"


# ── Data loaders ─────────────────────────────────────────────────────────────

def _safe(row, col, default):
    try:
        v = row.get(col, default)
        return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    except Exception:
        return default


def parse_brfss_row(row):
    """Convert one raw BRFSS row to 18-feature vector + label. Returns None to skip."""
    t = _safe(row, BRFSS_TARGET, None)
    if t not in (1, 2):
        return None
    label = 1 if t == 1 else 0

    def g(key, default=0):
        return _safe(row, BRFSS_MAP[key], default)

    age = min(80.0, max(18.0, float(g("age", 50))))
    sex = 1 if float(g("sex", 2)) == 1 else 0
    high_bp = 1 if float(g("high_bp", 1)) == 2 else 0
    high_chol = 1 if float(g("high_chol", 1)) == 2 else 0

    diab = float(g("diabetes", 3))
    diabetes = 1 if diab == 1 else 0

    bmi = float(g("bmi_cat", 2))
    bmi_cat = int(bmi) if 1 <= bmi <= 4 else 2

    smk = float(g("smoked_ever", 4))
    smoked_ever = 0 if smk == 4 else 1
    exercise = 1 if float(g("exercise", 1)) == 1 else 0
    binge = 1 if float(g("binge_drink", 1)) == 2 else 0
    low_frvt = 1 if float(g("low_frvt", 2)) == 1 else 0
    low_veg = 1 if float(g("low_veg", 2)) == 1 else 0

    slp = float(g("sleep_hrs", 7))
    sleep_hrs = min(24.0, max(1.0, slp))

    mh = float(g("mental_hlth", 88))
    mental_hlth = 0.0 if mh == 88 else min(30.0, max(0.0, mh))

    stroke = 1 if float(g("stroke_current", 2)) == 1 else 0

    gh = float(g("gen_health", 3))
    gen_health = int(gh) if 1 <= gh <= 5 else 3

    ph = float(g("phys_hlth", 88))
    phys_hlth = 0.0 if ph == 88 else min(30.0, max(0.0, ph))

    fh = float(g("family_history", 2))
    family_history = 1 if fh == 1 else 0

    st = float(g("stress", 1))
    stress = int(st) - 1 if 1 <= st <= 3 else 0

    return [age, sex, high_bp, high_chol, diabetes, bmi_cat, smoked_ever,
            exercise, binge, low_frvt, low_veg, sleep_hrs, mental_hlth,
            stroke, gen_health, phys_hlth, family_history, stress, label]


def load_brfss(path):
    print(f"Loading BRFSS 2022 ({path}) — may take 1-2 min...")
    df = pd.read_sas(path, format="xport", encoding="latin-1")
    print(f"  Raw rows: {len(df):,}  cols: {len(df.columns)}")

    # Check target
    if BRFSS_TARGET not in df.columns:
        alts = [c for c in df.columns if "MICHD" in c or "CVDCRHD" in c]
        if not alts:
            raise ValueError("Cannot find heart-disease target column in BRFSS data.")
        print(f"  Using target column: {alts[0]}")
        df = df.rename(columns={alts[0]: BRFSS_TARGET})

    rows = []
    for _, row in df.iterrows():
        r = parse_brfss_row(row)
        if r is not None:
            rows.append(r)

    data = pd.DataFrame(rows, columns=FEATURES + ["target"])
    print(f"  Usable: {len(data):,}  disease rate: {data['target'].mean():.1%}")
    return data, True


def load_uci(path):
    print(f"Loading UCI Cleveland ({path})...")
    df = pd.read_csv(path, na_values=["?"])
    df.columns = df.columns.str.strip().str.lower()

    # Identify target
    target_col = next(
        (c for c in ["target", "num", "condition", df.columns[-1]] if c in df.columns),
        df.columns[-1]
    )
    y = (df[target_col].fillna(0) > 0).astype(int)

    trestbps = df.get("trestbps", pd.Series([120] * len(df))).fillna(120)
    chol     = df.get("chol",     pd.Series([200] * len(df))).fillna(200)
    fbs      = df.get("fbs",      pd.Series([0]   * len(df))).fillna(0)
    exang    = df.get("exang",    pd.Series([0]   * len(df))).fillna(0)
    oldpeak  = df.get("oldpeak",  pd.Series([0.0] * len(df))).fillna(0)

    data = pd.DataFrame({
        "age":            df.get("age", pd.Series([50]*len(df))).fillna(50),
        "sex":            df.get("sex", pd.Series([1]*len(df))).fillna(1),
        "high_bp":        (trestbps > 130).astype(int),
        "high_chol":      (chol > 200).astype(int),
        "diabetes":       fbs.astype(int),
        "bmi_cat":        2,
        "smoked_ever":    0,
        "exercise":       (exang == 0).astype(int),
        "binge_drink":    0,
        "low_frvt":       0,
        "low_veg":        0,
        "sleep_hrs":      7,
        "mental_hlth":    0,
        "stroke_current": 0,
        "gen_health":     3,
        "phys_hlth":      (oldpeak * 3).clip(0, 30).astype(int),
        "family_history": 0,
        "stress":         0,
        "target":         y,
    })
    data = data.dropna()
    print(f"  Rows: {len(data):,}  disease rate: {data['target'].mean():.1%}")
    return data, False


def load_data():
    brfss = os.path.join(BASE, "LLCP2022.XPT")
    uci   = os.path.join(BASE, "heart.csv")
    if os.path.exists(brfss):
        return load_brfss(brfss)
    if os.path.exists(uci):
        return load_uci(uci)
    raise FileNotFoundError(
        "No dataset found.\n"
        "  Option A (recommended): Place 'LLCP2022.XPT' (CDC BRFSS 2022) in this folder.\n"
        "  Option B: Place 'heart.csv' (UCI Cleveland) in this folder."
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(data, using_brfss):
    from sklearn.ensemble import (RandomForestClassifier,
                                   GradientBoostingClassifier,
                                   VotingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                          train_test_split, GridSearchCV)
    from sklearn.metrics import (roc_auc_score, f1_score,
                                  confusion_matrix, accuracy_score)

    X = data[FEATURES].values.astype(float)
    y = data["target"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTuning Random Forest...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)
    rf_grid = GridSearchCV(rf, {
        "n_estimators": [200],
        "max_depth":    [4, 5],
        "min_samples_leaf": [5, 10],
    }, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"  Best RF params: {rf_grid.best_params_}")

    print("Tuning Gradient Boosting...")
    gbm = GradientBoostingClassifier(random_state=42)
    gbm_grid = GridSearchCV(gbm, {
        "n_estimators":   [100, 200],
        "max_depth":      [3, 4],
        "learning_rate":  [0.05, 0.1],
    }, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0)
    gbm_grid.fit(X_train, y_train)
    best_gbm = gbm_grid.best_estimator_
    print(f"  Best GBM params: {gbm_grid.best_params_}")

    lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1)

    voting = VotingClassifier(
        estimators=[("rf", best_rf), ("gbm", best_gbm), ("lr", lr)],
        voting="soft",
        weights=[2, 2, 1],
    )

    print("\nRunning 10-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(voting, Xs, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    print("Fitting final model on full training set...")
    voting.fit(X_train, y_train)

    # Individual model scores for all_results
    all_results = {}
    for name, clf in [("Random Forest", best_rf), ("Gradient Boosting", best_gbm),
                      ("Logistic Regression", lr)]:
        clf.fit(X_train, y_train)
        s = cross_val_score(clf, X_train, y_train, cv=3, scoring="roc_auc")
        all_results[name] = {"auc": round(s.mean(), 3)}

    # Test metrics
    y_pred  = voting.predict(X_test)
    y_proba = voting.predict_proba(X_test)[:, 1]
    train_acc = accuracy_score(y_train, voting.predict(X_train))
    test_acc  = accuracy_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_proba)
    f1        = f1_score(y_test, y_pred)
    cm        = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    overfit_gap = round((train_acc - test_acc) * 100, 1)

    oob = getattr(best_rf, "oob_score_", 0)

    print(f"\n{'='*50}")
    print(f"  Train Accuracy : {train_acc*100:.1f}%")
    print(f"  Test Accuracy  : {test_acc*100:.1f}%")
    print(f"  ROC-AUC        : {auc:.3f}")
    print(f"  F1 Score       : {f1:.3f}")
    print(f"  Sensitivity    : {sensitivity:.3f}")
    print(f"  Specificity    : {specificity:.3f}")
    print(f"  Overfit Gap    : {overfit_gap}%")
    print(f"  OOB Score (RF) : {oob*100:.1f}%")
    print(f"{'='*50}\n")

    save_data = {
        "model":         voting,
        "scaler":        scaler,
        "algorithm":     "Voting Ensemble (RF + GBM + LR)",
        "dataset":       "CDC BRFSS 2022" if using_brfss else "UCI Cleveland",
        "using_brfss":   using_brfss,
        "age_min":       18,
        "age_max":       80,
        "cv_accuracy":   round(cv_scores.mean() * 100, 1),
        "cv_std":        round(cv_scores.std() * 100, 1),
        "test_accuracy": round(test_acc * 100, 1),
        "auc_score":     round(auc, 3),
        "f1_score":      round(f1, 3),
        "sensitivity":   round(sensitivity, 3),
        "specificity":   round(specificity, 3),
        "oob_score":     round(oob * 100, 1),
        "overfit_gap":   overfit_gap,
        "n_total":       len(data),
        "all_results":   all_results,
        "best_params":   {
            "rf":  rf_grid.best_params_,
            "gbm": gbm_grid.best_params_,
        },
    }

    out = os.path.join(BASE, "heart_model.pkl")
    joblib.dump(save_data, out)
    print(f"Model saved -> {out}")
    return save_data


if __name__ == "__main__":
    try:
        data, using_brfss = load_data()
        train(data, using_brfss)
        print("Done! Run:  python heart_health_app.py")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
