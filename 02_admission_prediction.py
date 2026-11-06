"""
CT-264 Programming for AI - Complex Computing Activity (CCA)
AI-Based Patient Admission Prediction System

Group members:
    CT-24033  Muhammad Hamza Siddiqui
    CT-24044  Saif Ur Rehman
    CT-24312  Muhammad Talha Bin Suhail

Goal: predict, at the moment of emergency department (ED) triage, whether a
patient will be ADMITTED to the hospital or DISCHARGED, so the hospital can plan
beds and staff earlier and reduce ED crowding.

Pipeline:
    1. Load the two datasets
    2. Data understanding and quality report
    3. Preprocessing (inconsistencies, outliers, missing values, encoding)
    4. Exploratory Data Analysis (EDA) with statistics and charts
    5. Class imbalance check and handling
    6. Model training: Logistic Regression, Decision Tree, Random Forest,
       Gradient Boosting
    7. Hyperparameter tuning + soft-voting ensemble (performance improvement)
    8. Evaluation: accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
       confusion matrices, ROC/PR curves, threshold tuning, feature importance
    9. Dataset 2 (Korea KTAS): cleaning, EDA and its own model
   10. Cross-dataset (external) validation with shared triage features
   11. Hospital operations view: predicted vs actual admissions per time slot

Run:  python 02_admission_prediction.py
All charts are saved in outputs/figures, tables in outputs/metrics and the
trained model in outputs/models (used by the GUI 03_admission_app.py).
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # save charts to files without opening windows
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                     StratifiedKFold, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              VotingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve, classification_report,
                             balanced_accuracy_score)

from admission_utils import (TARGET, CATEGORICAL_FEATURES, NUMERIC_FEATURES, FEATURES,
                             LEAKAGE_SUSPECTS, calibrate_probability,
                             clean_patient_flow, clean_ktas, harmonise_patient_flow,
                             harmonise_ktas, KTAS_DISPOSITION_NAMES, KTAS_ARRIVAL_NAMES,
                             COLOR_ADMIT, COLOR_DISCHARGE, MODEL_COLORS, TEXT_SECONDARY,
                             GRID_COLOR)

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")   # Korean text in Dataset 2 prints correctly
except AttributeError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
METRIC_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
for folder in (FIG_DIR, METRIC_DIR, MODEL_DIR):
    os.makedirs(folder, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------- chart style
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.edgecolor": GRID_COLOR, "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "axes.grid": True, "grid.color": GRID_COLOR, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
OUTCOME_PALETTE = {"Discharged": COLOR_DISCHARGE, "Admitted": COLOR_ADMIT}


def save_fig(name):
    plt.tight_layout()
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   [figure saved] outputs/figures/{name}")


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def bar_labels(ax, fmt="{:.1f}%", inside=False):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, fontsize=8, color=TEXT_SECONDARY,
                     padding=2, label_type="center" if inside else "edge")


start_time = time.time()

# =============================================================================
# 1. LOAD DATA
# =============================================================================
section("1. LOADING THE DATASETS")
flow_raw = pd.read_csv(os.path.join(DATA_DIR, "ed_patient_flow.csv"))
ktas_raw = pd.read_csv(os.path.join(DATA_DIR, "ktas_triage.csv"), encoding="utf-8-sig")

print(f"Dataset 1 - ED Patient Flow (USA): {flow_raw.shape[0]:,} visits x {flow_raw.shape[1]} columns")
print(f"Dataset 2 - KTAS Triage (Korea)  : {ktas_raw.shape[0]:,} visits x {ktas_raw.shape[1]} columns")
print("\nFirst 5 rows of Dataset 1:")
print(flow_raw.head().to_string())

# =============================================================================
# 2. DATA UNDERSTANDING & QUALITY REPORT
# =============================================================================
section("2. DATA UNDERSTANDING AND QUALITY REPORT (Dataset 1)")
print("\nData types:")
print(flow_raw.dtypes.value_counts().to_string())

print(f"\nDuplicate rows (ignoring visit_id): {flow_raw.drop(columns='visit_id').duplicated().sum()}")

missing = flow_raw.isna().mean().mul(100).sort_values(ascending=False)
missing = missing[missing > 0]
print("\nMissing values (% of rows):")
print(missing.round(2).to_string())

print("\nPlaceholder answers that actually mean 'missing':")
for col in ["gender", "ethnicity", "race", "marital_status", "employment_status"]:
    n = flow_raw[col].isin(["Unknown", "Patient Refused"]).sum()
    if n:
        print(f"   {col:<20}: {n:,} rows are 'Unknown' / 'Patient Refused'")

print("\nInconsistent spelling in arrival_mode:", sorted(flow_raw["arrival_mode"].dropna().unique()))

print("\nVital-sign ranges BEFORE cleaning:")
vitals_raw = ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate",
              "oxygen_saturation", "temperature_f"]
print(flow_raw[vitals_raw].describe().T[["min", "50%", "max"]].round(1).to_string())

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(missing.index[::-1], missing.values[::-1], color=COLOR_DISCHARGE, height=0.6)
for y, v in enumerate(missing.values[::-1]):
    ax.text(v + 0.5, y, f"{v:.1f}%", va="center", fontsize=8, color=TEXT_SECONDARY)
ax.set_xlabel("Missing values (% of visits)")
ax.set_title("Dataset 1: missing values per column (before cleaning)")
ax.grid(axis="y", visible=False)
save_fig("fig01_missing_values.png")

# =============================================================================
# 3. PREPROCESSING
# =============================================================================
section("3. PREPROCESSING")
flow, TOP_COMPLAINTS = clean_patient_flow(flow_raw)

print("Steps applied:")
print("  a) 'Unknown' / 'Patient Refused' -> missing value")
print("  b) Spelling unified: 'ambulance' -> 'Ambulance', ED site codes -> 'Site A/B/C'")
print("  c) Temperature converted from Fahrenheit to Celsius")
print("  d) Impossible vital signs (e.g. heart rate > 250) -> missing value")
print(f"  e) Rare chief complaints grouped into 'other' (kept top {len(TOP_COMPLAINTS)})")
print("  f) Target created: admitted = 1 if disposition == 'Admit' else 0")
print("  g) Missing numbers -> median + 'was missing' indicator; missing text -> 'Missing'")
print("  h) Text columns -> one-hot encoding; numbers -> standard scaling")
print(f"  i) Leakage check: {LEAKAGE_SUSPECTS} kept for EDA but excluded from the model")

removed = {c: int(flow[c].isna().sum() - flow_raw[c if c != 'temperature_c' else 'temperature_f'].isna().sum())
           for c in ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate",
                     "oxygen_saturation", "temperature_c"]}
print("\nImpossible vital-sign values set to missing:", removed)
n_before = len(flow)
flow = flow.drop_duplicates(subset=[c for c in flow.columns if c != "visit_id"])
print(f"Duplicate visits removed: {n_before - len(flow)}")
print("\nVital-sign ranges AFTER cleaning:")
vitals = ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate",
          "oxygen_saturation", "temperature_c"]
print(flow[vitals].describe().T[["min", "50%", "max"]].round(1).to_string())

flow["outcome"] = flow[TARGET].map({0: "Discharged", 1: "Admitted"})

# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS
# =============================================================================
section("4. EXPLORATORY DATA ANALYSIS (Dataset 1)")

print("\nStatistical summary of numeric features:")
print(flow[["age", "esi_level", "prior_ed_visits", "prior_admissions", "num_comorbidities",
            "home_medications"] + vitals].describe().T.round(2).to_string())

print("\nMean of key features by outcome:")
group_means = flow.groupby("outcome")[["age", "esi_level", "prior_admissions",
                                       "num_comorbidities", "heart_rate",
                                       "oxygen_saturation", "temperature_c"]].mean().round(2)
print(group_means.T.to_string())
group_means.T.to_csv(os.path.join(METRIC_DIR, "eda_means_by_outcome.csv"))

# 4.1 Target distribution
counts = flow["outcome"].value_counts()
fig, ax = plt.subplots(figsize=(6, 3.8))
bars = ax.bar(counts.index, counts.values, width=0.55,
              color=[OUTCOME_PALETTE[i] for i in counts.index])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / counts.sum():.1%})",
            ha="center", va="bottom", fontsize=9, color=TEXT_SECONDARY)
ax.set_ylim(0, counts.max() * 1.2)
ax.set_ylabel("Number of ED visits")
ax.set_title("Dataset 1: admitted vs discharged visits")
ax.grid(axis="x", visible=False)
save_fig("fig02_class_distribution.png")

# 4.2 Age distribution by outcome
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data=flow, x="age", hue="outcome", bins=40, stat="density",
             common_norm=False, element="step", fill=True, alpha=0.25,
             palette=OUTCOME_PALETTE, ax=ax, linewidth=1.5)
ax.set_xlabel("Age (years)")
ax.set_title("Age distribution: admitted patients are older")
save_fig("fig03_age_by_outcome.png")


def admission_rate_chart(col, title, fname, order=None, horizontal=False, figsize=(7, 4)):
    rate = flow.groupby(col)[TARGET].agg(["mean", "size"])
    rate["mean"] *= 100
    if order is not None:
        rate = rate.reindex(order)
    else:
        rate = rate.sort_values("mean")
    fig, ax = plt.subplots(figsize=figsize)
    labels = [str(int(i)) if isinstance(i, float) and i.is_integer() else str(i) for i in rate.index]
    if horizontal:
        ax.barh(labels, rate["mean"], color=COLOR_ADMIT, height=0.65)
        for y, v in enumerate(rate["mean"]):
            ax.text(v + 0.5, y, f"{v:.1f}%", va="center", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlabel("Admission rate (%)")
        ax.grid(axis="y", visible=False)
    else:
        ax.bar(labels, rate["mean"], color=COLOR_ADMIT, width=0.6)
        bar_labels(ax)
        ax.set_ylabel("Admission rate (%)")
        ax.grid(axis="x", visible=False)
    ax.set_title(title)
    save_fig(fname)
    return rate


# 4.3 Admission rate by triage level (ESI)
esi_rate = admission_rate_chart("esi_level", "Admission rate by triage level (ESI 1 = most urgent)",
                                "fig04_admission_by_esi.png", order=[1.0, 2.0, 3.0, 4.0, 5.0])
print("\nAdmission rate by ESI level (%):")
print(esi_rate.round(1).to_string())

# 4.4 Arrival mode
arr_rate = admission_rate_chart("arrival_mode", "Admission rate by arrival mode",
                                "fig05_admission_by_arrival.png", horizontal=True)
print("\nAdmission rate by arrival mode (%):")
print(arr_rate.round(1).to_string())

# 4.5 Chief complaint
cc_rate = admission_rate_chart("chief_complaint", "Admission rate by chief complaint (top complaints)",
                               "fig06_admission_by_complaint.png", horizontal=True, figsize=(8, 7.5))

# 4.6 Vital signs by outcome
fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
names = {"heart_rate": "Heart rate (bpm)", "systolic_bp": "Systolic BP (mmHg)",
         "diastolic_bp": "Diastolic BP (mmHg)", "respiratory_rate": "Respiratory rate (/min)",
         "oxygen_saturation": "Oxygen saturation (%)", "temperature_c": "Temperature (°C)"}
for ax, col in zip(axes.ravel(), vitals):
    sns.boxplot(data=flow, x="outcome", y=col, hue="outcome", palette=OUTCOME_PALETTE,
                order=["Discharged", "Admitted"], ax=ax, width=0.5, fliersize=1,
                linewidth=1, legend=False)
    ax.set_title(names[col], fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
fig.suptitle("Triage vital signs by outcome", fontweight="bold")
save_fig("fig07_vitals_by_outcome.png")

# 4.7 Correlation heatmap
corr_cols = ["admitted", "age", "esi_level", "prior_ed_visits", "prior_admissions",
             "num_comorbidities", "home_medications", "prior_imaging_tests",
             "heart_failure", "kidney_disease"] + vitals
corr = flow[corr_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
            annot_kws={"size": 7}, linewidths=0.5, linecolor="white", ax=ax,
            cbar_kws={"shrink": 0.7, "label": "Pearson correlation"})
ax.set_title("Correlation between numeric features and admission")
ax.grid(False)
save_fig("fig08_correlation_heatmap.png")
print("\nCorrelation of each numeric feature with 'admitted':")
print(corr["admitted"].drop("admitted").sort_values(ascending=False).round(3).to_string())

# 4.8 Patient flow heatmap: admissions by weekday and arrival time
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
hours = ["23-02", "03-06", "07-10", "11-14", "15-18", "19-22"]
pivot = flow.pivot_table(index="arrival_day", columns="arrival_hour_bin", values=TARGET,
                         aggfunc="sum").reindex(index=days, columns=hours)
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".0f", linewidths=1, linecolor="white",
            cbar_kws={"label": "Number of admissions"}, ax=ax)
ax.set_title("Patient flow: admissions by weekday and arrival time")
ax.set_xlabel("Arrival time (hour band)")
ax.set_ylabel("")
ax.grid(False)
save_fig("fig09_admissions_heatmap_day_hour.png")

# 4.9 History: prior admissions and comorbidities
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
prior = flow["prior_admissions"].clip(upper=5).map(lambda x: "5+" if x >= 5 else str(int(x)))
r1 = flow.groupby(prior)[TARGET].mean().mul(100).reindex(["0", "1", "2", "3", "4", "5+"])
axes[0].bar(r1.index, r1.values, color=COLOR_ADMIT, width=0.6)
axes[0].set_xlabel("Admissions in the previous year")
axes[0].set_ylabel("Admission rate (%)")
axes[0].set_title("More previous admissions -> higher risk")
bar_labels(axes[0])
axes[0].grid(axis="x", visible=False)
comorb = pd.cut(flow["num_comorbidities"], [-1, 0, 5, 10, 20, 300],
                labels=["0", "1-5", "6-10", "11-20", "21+"])
r2 = flow.groupby(comorb, observed=False)[TARGET].mean().mul(100)
axes[1].bar(r2.index.astype(str), r2.values, color=COLOR_ADMIT, width=0.6)
axes[1].set_xlabel("Number of past medical conditions")
axes[1].set_title("More comorbidities -> higher risk")
bar_labels(axes[1])
axes[1].grid(axis="x", visible=False)
save_fig("fig10_history_vs_admission.png")

# 4.10 Insurance & age group
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
age_grp = pd.cut(flow["age"], [17, 30, 45, 60, 75, 120], labels=["18-30", "31-45", "46-60", "61-75", "76+"])
r3 = flow.groupby(age_grp, observed=False)[TARGET].mean().mul(100)
axes[0].bar(r3.index.astype(str), r3.values, color=COLOR_ADMIT, width=0.6)
axes[0].set_title("Admission rate by age group")
axes[0].set_ylabel("Admission rate (%)")
bar_labels(axes[0])
axes[0].grid(axis="x", visible=False)
r4 = flow.groupby("insurance_status")[TARGET].mean().mul(100).sort_values()
axes[1].bar(r4.index, r4.values, color=COLOR_ADMIT, width=0.6)
axes[1].set_title("Admission rate by insurance type")
bar_labels(axes[1])
axes[1].grid(axis="x", visible=False)
save_fig("fig11_age_insurance.png")

# =============================================================================
# 5. CLASS IMBALANCE
# =============================================================================
section("5. CLASS IMBALANCE CHECK")
n_pos, n_neg = int(flow[TARGET].sum()), int((1 - flow[TARGET]).sum())
print(f"Admitted  : {n_pos:,} ({n_pos / len(flow):.1%})")
print(f"Discharged: {n_neg:,} ({n_neg / len(flow):.1%})")
print(f"Imbalance ratio: {n_neg / n_pos:.2f} discharged for every admitted patient")
print("-> Moderate imbalance. A model that always says 'discharge' would already be")
print(f"   {n_neg / len(flow):.1%} accurate, so accuracy alone is misleading.")
print("Handling: (1) stratified train/test split, (2) class_weight='balanced' so")
print("   mistakes on admitted patients cost more, (3) metrics that focus on the")
print("   minority class (recall, F1, PR-AUC), (4) decision-threshold tuning.")

# =============================================================================
# 6. TRAIN / TEST SPLIT AND MODELS
# =============================================================================
section("6. MODEL TRAINING")
X = flow[FEATURES]
y = flow[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"Training set: {len(X_train):,} visits | Test set: {len(X_test):,} visits")
print(f"Admission rate - train {y_train.mean():.3f}, test {y_test.mean():.3f} (stratified)")


def make_preprocessor(numeric_features=NUMERIC_FEATURES):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_features),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


base_models = {
    "Logistic Regression (no balancing)": LogisticRegression(max_iter=2000),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=50,
                                            class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=5,
                                            class_weight="balanced", n_jobs=-1,
                                            random_state=RANDOM_STATE),
    "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced",
                                                        random_state=RANDOM_STATE),
}

fitted = {}
for name, model in base_models.items():
    t0 = time.time()
    pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    print(f"   trained {name:<38} in {time.time() - t0:5.1f} s")

# =============================================================================
# 7. PERFORMANCE IMPROVEMENT: TUNING + ENSEMBLE
# =============================================================================
section("7. PERFORMANCE IMPROVEMENT (hyperparameter tuning + ensemble)")
param_dist = {
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
    "model__max_iter": [200, 300, 400, 600],
    "model__max_leaf_nodes": [15, 31, 63, 127],
    "model__min_samples_leaf": [20, 50, 100, 200],
    "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
}
search = RandomizedSearchCV(
    Pipeline([("prep", make_preprocessor()),
              ("model", HistGradientBoostingClassifier(class_weight="balanced",
                                                       random_state=RANDOM_STATE))]),
    param_distributions=param_dist, n_iter=12, scoring="roc_auc",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1, random_state=RANDOM_STATE, verbose=0)
t0 = time.time()
search.fit(X_train, y_train)
print(f"RandomizedSearchCV: 12 settings x 3 folds in {time.time() - t0:.0f} s")
print("Best parameters :", {k.replace('model__', ''): v for k, v in search.best_params_.items()})
print(f"Best CV ROC-AUC : {search.best_score_:.4f}")
fitted["Tuned Gradient Boosting"] = search.best_estimator_
pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]] \
    .sort_values("rank_test_score").to_csv(os.path.join(METRIC_DIR, "tuning_results.csv"), index=False)

best_hgb = search.best_estimator_.named_steps["model"]

# 7b. Data-leakage check: what happens if the home-medication count is added?
leak_cols = NUMERIC_FEATURES + LEAKAGE_SUSPECTS
leak_model = Pipeline([("prep", make_preprocessor(leak_cols)),
                       ("model", HistGradientBoostingClassifier(**best_hgb.get_params()))])
leak_model.fit(flow.loc[X_train.index, CATEGORICAL_FEATURES + leak_cols], y_train)
leak_auc = roc_auc_score(y_test, leak_model.predict_proba(
    flow.loc[X_test.index, CATEGORICAL_FEATURES + leak_cols])[:, 1])
final_auc = roc_auc_score(y_test, search.best_estimator_.predict_proba(X_test)[:, 1])
print("\nData-leakage check (tuned Gradient Boosting):")
print(f"   with home_medications    : test ROC-AUC = {leak_auc:.4f}")
print(f"   without (final features) : test ROC-AUC = {final_auc:.4f}")
med_means = flow.groupby(TARGET)["home_medications"].mean()
print(f"   Admitted patients list {med_means[1]:.1f} home medications vs {med_means[0]:.1f} for discharged")
print("   patients, because the list is completed during admission (medication reconciliation).")
print("   It is not reliably known at triage, so it is excluded from the final model.")
pd.DataFrame({"Feature set": ["With home_medications (leaky)", "Final model (without)"],
              "ROC-AUC": [leak_auc, final_auc]}).round(4) \
    .to_csv(os.path.join(METRIC_DIR, "leakage_check.csv"), index=False)
ensemble = Pipeline([
    ("prep", make_preprocessor()),
    ("model", VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=5,
                                          class_weight="balanced", n_jobs=-1,
                                          random_state=RANDOM_STATE)),
            ("gb", HistGradientBoostingClassifier(**best_hgb.get_params())),
        ],
        voting="soft", weights=[1, 1, 2])),
])
t0 = time.time()
ensemble.fit(X_train, y_train)
fitted["Voting Ensemble"] = ensemble
print(f"Soft-voting ensemble (LR + RF + tuned GB, weights 1:1:2) trained in {time.time() - t0:.0f} s")

# =============================================================================
# 8. EVALUATION
# =============================================================================
section("8. EVALUATION ON THE UNSEEN TEST SET")


def evaluate(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred),
        "Recall (sensitivity)": recall_score(y_true, pred),
        "Specificity": tn / (tn + fp),
        "F1-score": f1_score(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, prob),
        "PR-AUC": average_precision_score(y_true, prob),
    }


probs, results = {}, {}
for name, pipe in fitted.items():
    probs[name] = pipe.predict_proba(X_test)[:, 1]
    results[name] = evaluate(y_test, probs[name])
results_df = pd.DataFrame(results).T.round(4)
results_df.index.name = "Model"
results_df.to_csv(os.path.join(METRIC_DIR, "model_comparison.csv"))
print(results_df.to_string())

print("\nEffect of class balancing on Logistic Regression:")
for n in ["Logistic Regression (no balancing)", "Logistic Regression"]:
    print(f"   {n:<38} recall={results[n]['Recall (sensitivity)']:.3f}  "
          f"precision={results[n]['Precision']:.3f}  F1={results[n]['F1-score']:.3f}")

main_models = [m for m in fitted if m != "Logistic Regression (no balancing)"]
best_name = max(main_models, key=lambda m: results[m]["ROC-AUC"])
best_model = fitted[best_name]
print(f"\nBest model by ROC-AUC: {best_name} (ROC-AUC = {results[best_name]['ROC-AUC']:.4f})")

# 8.1 Model comparison chart
metric_cols = ["Accuracy", "Precision", "Recall (sensitivity)", "F1-score", "ROC-AUC"]
plot_df = results_df.loc[main_models, metric_cols]
fig, ax = plt.subplots(figsize=(11, 4.8))
width = 0.8 / len(main_models)
x = np.arange(len(metric_cols))
for i, m in enumerate(main_models):
    ax.bar(x + i * width - 0.4 + width / 2, plot_df.loc[m], width=width * 0.92,
           color=MODEL_COLORS[i], label=m)
ax.set_xticks(x)
ax.set_xticklabels(metric_cols)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Score (test set)")
ax.set_title("Model comparison on the test set (threshold 0.5)")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
ax.grid(axis="x", visible=False)
save_fig("fig12_model_comparison.png")

# 8.2 ROC and Precision-Recall curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, m in enumerate(main_models):
    fpr, tpr, _ = roc_curve(y_test, probs[m])
    axes[0].plot(fpr, tpr, lw=2, color=MODEL_COLORS[i], label=f"{m} (AUC {results[m]['ROC-AUC']:.3f})")
    prec, rec, _ = precision_recall_curve(y_test, probs[m])
    axes[1].plot(rec, prec, lw=2, color=MODEL_COLORS[i], label=f"{m} (AP {results[m]['PR-AUC']:.3f})")
axes[0].plot([0, 1], [0, 1], ls="--", color="#a3a29d", lw=1, label="Random guess")
axes[0].set_xlabel("False positive rate (1 - specificity)")
axes[0].set_ylabel("True positive rate (recall)")
axes[0].set_title("ROC curves")
axes[0].legend(fontsize=8, loc="lower right")
axes[1].axhline(y_test.mean(), ls="--", color="#a3a29d", lw=1, label=f"Baseline ({y_test.mean():.2f})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall curves")
axes[1].legend(fontsize=8, loc="lower left")
save_fig("fig13_roc_pr_curves.png")

# 8.3 Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, m in zip(axes.ravel(), main_models):
    cm = confusion_matrix(y_test, (probs[m] >= 0.5).astype(int))
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{cm[r, c]:,}\n({pct[r, c]:.1f}%)" for c in range(2)] for r in range(2)])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Discharge", "Admit"], yticklabels=["Discharge", "Admit"],
                linewidths=1, linecolor="white", annot_kws={"size": 10})
    ax.set_title(m, fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
fig.suptitle("Confusion matrices (test set, 20,000 visits)", fontweight="bold")
save_fig("fig14_confusion_matrices.png")

print(f"\nClassification report - {best_name}:")
print(classification_report(y_test, (probs[best_name] >= 0.5).astype(int),
                            target_names=["Discharge", "Admit"], digits=3))

# 8.4 Threshold tuning on the best model
thresholds = np.arange(0.10, 0.91, 0.02)
rows = []
for t in thresholds:
    pred = (probs[best_name] >= t).astype(int)
    rows.append({"threshold": round(t, 2),
                 "precision": precision_score(y_test, pred, zero_division=0),
                 "recall": recall_score(y_test, pred),
                 "f1": f1_score(y_test, pred)})
thr_df = pd.DataFrame(rows)
best_thr = float(thr_df.loc[thr_df["f1"].idxmax(), "threshold"])
thr_df.to_csv(os.path.join(METRIC_DIR, "threshold_tuning.csv"), index=False)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(thr_df["threshold"], thr_df["precision"], lw=2, color=MODEL_COLORS[0], label="Precision")
ax.plot(thr_df["threshold"], thr_df["recall"], lw=2, color=MODEL_COLORS[1], label="Recall")
ax.plot(thr_df["threshold"], thr_df["f1"], lw=2, color=MODEL_COLORS[2], label="F1-score")
ax.axvline(best_thr, ls="--", color="#a3a29d", lw=1)
ax.text(best_thr + 0.01, 0.05, f"best F1 threshold = {best_thr:.2f}", fontsize=8, color=TEXT_SECONDARY)
ax.set_xlabel("Decision threshold (probability of admission)")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title(f"Threshold tuning - {best_name}")
ax.legend()
save_fig("fig15_threshold_tuning.png")
tuned_metrics = evaluate(y_test, probs[best_name], best_thr)
print(f"Threshold with best F1: {best_thr:.2f}")
print("Metrics at 0.50       :", {k: round(v, 3) for k, v in results[best_name].items()})
print(f"Metrics at {best_thr:.2f}       :", {k: round(v, 3) for k, v in tuned_metrics.items()})

# 8.5 Feature importance (permutation importance on the original columns)
print("\nComputing permutation feature importance (5,000 test visits, 5 repeats)...")
sample_idx = X_test.sample(5000, random_state=RANDOM_STATE).index
perm = permutation_importance(best_model, X_test.loc[sample_idx], y_test.loc[sample_idx],
                              scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean,
                    "std": perm.importances_std}).sort_values("importance", ascending=False)
imp.to_csv(os.path.join(METRIC_DIR, "feature_importance.csv"), index=False)
print(imp.head(15).round(4).to_string(index=False))
top = imp.head(15)[::-1]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(top["feature"], top["importance"], xerr=top["std"], color=COLOR_DISCHARGE, height=0.6,
        error_kw={"ecolor": TEXT_SECONDARY, "lw": 1})
ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
ax.set_title(f"Top 15 most important features - {best_name}")
ax.grid(axis="y", visible=False)
save_fig("fig16_feature_importance.png")

# =============================================================================
# 9. DATASET 2: KOREAN TRIAGE (KTAS)
# =============================================================================
section("9. DATASET 2 - KOREAN TRIAGE AND ACUITY SCALE (KTAS)")
print("Raw disposition codes:")
print(ktas_raw["Disposition"].map(KTAS_DISPOSITION_NAMES).value_counts().to_string())
for col in ["SBP", "DBP", "HR", "RR", "BT", "Saturation"]:
    bad = ktas_raw[col].astype(str).str.contains("측불").sum()
    print(f"   {col:<11}: {bad:3d} values are the text '측불' (Korean for 'unmeasurable')")
print(f"   NRS_pain   : {ktas_raw['NRS_pain'].isna().sum()} missing")

ktas = clean_ktas(ktas_raw)
print(f"\nAfter cleaning: {len(ktas):,} visits (transfers and deaths removed)")
print(f"Admitted (ward / ICU / operation): {ktas[TARGET].sum()} ({ktas[TARGET].mean():.1%})")
print("Missing values after converting text to numbers:")
print(ktas[["SBP", "DBP", "HR", "RR", "BT", "Saturation", "NRS_pain"]].isna().sum().to_string())

ktas["outcome"] = ktas[TARGET].map({0: "Discharged", 1: "Admitted"})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
c = ktas["outcome"].value_counts()
axes[0].bar(c.index, c.values, color=[OUTCOME_PALETTE[i] for i in c.index], width=0.55)
for i, v in enumerate(c.values):
    axes[0].text(i, v, f"{v} ({v / c.sum():.1%})", ha="center", va="bottom", fontsize=9,
                 color=TEXT_SECONDARY)
axes[0].set_title("KTAS: class distribution")
axes[0].grid(axis="x", visible=False)
k_rate = ktas.groupby("KTAS_RN")[TARGET].mean().mul(100)
axes[1].bar(k_rate.index.astype(str), k_rate.values, color=COLOR_ADMIT, width=0.6)
bar_labels(axes[1])
axes[1].set_xlabel("Nurse triage level (KTAS 1 = most urgent)")
axes[1].set_ylabel("Admission rate (%)")
axes[1].set_title("KTAS: admission rate by triage level")
axes[1].grid(axis="x", visible=False)
sns.boxplot(data=ktas, x="outcome", y="Age", hue="outcome", palette=OUTCOME_PALETTE,
            order=["Discharged", "Admitted"], ax=axes[2], width=0.5, legend=False)
axes[2].set_title("KTAS: age by outcome")
axes[2].set_xlabel("")
save_fig("fig17_ktas_eda.png")

ktas_num = ["Age", "Patients number per hour", "NRS_pain", "SBP", "DBP", "HR", "RR", "BT",
            "Saturation", "KTAS_RN"]
ktas_cat = ["Group", "Sex", "Arrival mode", "Injury", "Mental", "Pain"]
print("\nMean of features by outcome (KTAS):")
print(ktas.groupby("outcome")[ktas_num].mean().round(2).T.to_string())

ktas_pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                      ("scale", StandardScaler())]), ktas_num),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ktas_cat),
])
ktas_models = {
    "Nurse KTAS level only (baseline)": Pipeline([
        ("prep", ColumnTransformer([("k", StandardScaler(), ["KTAS_RN"])])),
        ("model", LogisticRegression(class_weight="balanced"))]),
    "Logistic Regression": Pipeline([("prep", ktas_pre),
                                     ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "Random Forest": Pipeline([("prep", ktas_pre),
                               ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                                class_weight="balanced",
                                                                random_state=RANDOM_STATE, n_jobs=-1))]),
    "Gradient Boosting": Pipeline([("prep", ktas_pre),
                                   ("model", HistGradientBoostingClassifier(
                                       max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                       class_weight="balanced", random_state=RANDOM_STATE))]),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
ktas_rows = {}
Xk, yk = ktas[ktas_num + ktas_cat], ktas[TARGET]
for name, pipe in ktas_models.items():
    s = cross_validate(pipe, Xk, yk, cv=cv,
                       scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    ktas_rows[name] = {m: s[f"test_{m}"].mean() for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    ktas_rows[name]["roc_auc_std"] = s["test_roc_auc"].std()
ktas_df = pd.DataFrame(ktas_rows).T.round(4)
ktas_df.columns = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "ROC-AUC std"]
ktas_df.index.name = "Model (5-fold CV)"
ktas_df.to_csv(os.path.join(METRIC_DIR, "ktas_model_comparison.csv"))
print("\nKTAS models - 5-fold stratified cross-validation:")
print(ktas_df.to_string())

# =============================================================================
# 10. CROSS-DATASET (EXTERNAL) VALIDATION
# =============================================================================
section("10. CROSS-DATASET VALIDATION (train on USA data -> test on Korean data)")
H_train = harmonise_patient_flow(flow.loc[X_train.index])
H_test = harmonise_patient_flow(flow.loc[X_test.index])
H_ktas = harmonise_ktas(ktas)
triage_model = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("model", HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
])
triage_model.fit(H_train, y_train)
ext = {
    "USA test set (same hospital system)": roc_auc_score(y_test, triage_model.predict_proba(H_test)[:, 1]),
    "Korea KTAS (external, never seen)": roc_auc_score(ktas[TARGET], triage_model.predict_proba(H_ktas)[:, 1]),
}
ext_df = pd.DataFrame({"ROC-AUC": ext}).round(4)
ext_df.index.name = "Evaluation data (10 shared triage features)"
ext_df.to_csv(os.path.join(METRIC_DIR, "external_validation.csv"))
print(ext_df.to_string())
print("-> The triage-only model keeps useful discrimination on a different country,")
print("   but loses accuracy: hospitals differ in patients, triage scales and admission")
print("   policies, so a model should be re-calibrated locally before real use.")

fig, ax = plt.subplots(figsize=(7, 5))
for (label, Xh, yh), col in zip([("USA test set", H_test, y_test), ("Korea KTAS (external)", H_ktas, ktas[TARGET])],
                                [MODEL_COLORS[0], MODEL_COLORS[1]]):
    p = triage_model.predict_proba(Xh)[:, 1]
    fpr, tpr, _ = roc_curve(yh, p)
    ax.plot(fpr, tpr, lw=2, color=col, label=f"{label} (AUC {roc_auc_score(yh, p):.3f})")
ax.plot([0, 1], [0, 1], ls="--", color="#a3a29d", lw=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("External validation of the triage-only model")
ax.legend(loc="lower right")
save_fig("fig18_external_validation.png")

# =============================================================================
# 11. HOSPITAL OPERATIONS VIEW
# =============================================================================
section("11. HOSPITAL OPERATIONS VIEW - expected bed demand")
ops = flow.loc[X_test.index, ["arrival_hour_bin", "arrival_day"]].copy()
ops["actual"] = y_test.values
PRIOR_RATIO = n_neg / n_pos
ops["raw_score_sum"] = probs[best_name]
ops["expected"] = calibrate_probability(probs[best_name], PRIOR_RATIO)
by_hour = ops.groupby("arrival_hour_bin")[["actual", "raw_score_sum", "expected"]].sum().reindex(hours)
print("Balanced class weights push scores upwards, so summing raw scores over-estimates")
print(f"admissions ({ops['raw_score_sum'].sum():.0f} vs {ops['actual'].sum()} actual). After prior")
print(f"correction (odds divided by {PRIOR_RATIO:.2f}) the forecast is {ops['expected'].sum():.0f} admissions.")
by_hour.round(0).astype(int).to_csv(os.path.join(METRIC_DIR, "bed_demand_by_hour.csv"))
print("Admissions in the test set per arrival time band (actual vs model-expected):")
print(by_hour.round(0).astype(int).to_string())
fig, ax = plt.subplots(figsize=(8, 4.2))
xh = np.arange(len(hours))
ax.bar(xh - 0.2, by_hour["actual"], width=0.38, color=MODEL_COLORS[0], label="Actual admissions")
ax.bar(xh + 0.2, by_hour["expected"], width=0.38, color=MODEL_COLORS[1], label="Forecast (sum of calibrated probabilities)")
ax.set_xticks(xh)
ax.set_xticklabels(hours)
ax.set_xlabel("Arrival time (hour band)")
ax.set_ylabel("Number of admissions (test set)")
ax.set_title("Bed-demand forecast: actual vs expected admissions")
ax.legend(fontsize=8)
ax.grid(axis="x", visible=False)
save_fig("fig19_bed_demand_forecast.png")

# =============================================================================
# 12. SAVE MODEL FOR THE GUI
# =============================================================================
section("12. SAVING THE FINAL MODEL")
categories = {c: sorted(flow[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}
defaults = {c: float(flow[c].median()) for c in NUMERIC_FEATURES}
defaults.update({c: flow[c].mode()[0] for c in CATEGORICAL_FEATURES})
joblib.dump(best_model, os.path.join(MODEL_DIR, "admission_model.joblib"))
X_test.assign(actual_admitted=y_test.values).sample(500, random_state=RANDOM_STATE) \
    .to_csv(os.path.join(MODEL_DIR, "demo_patients.csv"), index=False)
meta = {
    "model_name": best_name,
    "threshold": best_thr,
    "prior_ratio": PRIOR_RATIO,
    "test_metrics": {k: round(v, 4) for k, v in results[best_name].items()},
    "top_complaints": TOP_COMPLAINTS,
    "categories": categories,
    "defaults": defaults,
    "train_size": len(X_train),
    "test_size": len(X_test),
}
with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved outputs/models/admission_model.joblib  ({best_name})")
print("Saved outputs/models/model_info.json and demo_patients.csv")

# =============================================================================
# SUMMARY
# =============================================================================
section("PROJECT SUMMARY")
print(f"Best model           : {best_name}")
for k, v in results[best_name].items():
    print(f"   {k:<22}: {v:.4f}")
print(f"Best F1 threshold    : {best_thr:.2f}  (F1 = {tuned_metrics['F1-score']:.4f}, "
      f"recall = {tuned_metrics['Recall (sensitivity)']:.4f})")
print(f"Top 5 predictors     : {', '.join(imp['feature'].head(5))}")
print(f"KTAS best CV ROC-AUC : {ktas_df['ROC-AUC'].max():.4f} ({ktas_df['ROC-AUC'].idxmax()})")
print(f"External ROC-AUC     : {ext['Korea KTAS (external, never seen)']:.4f}")
print(f"\nTotal run time: {(time.time() - start_time) / 60:.1f} minutes")
print("Next step: run  python 03_admission_app.py  to open the prediction interface.")
