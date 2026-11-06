"""
Shared helpers for the Patient Admission Prediction project.
Used by 02_admission_prediction.py (training) and 03_admission_app.py (GUI),
so both clean a patient record in exactly the same way.
"""

import numpy as np
import pandas as pd

TARGET = "admitted"

# Columns that are cleaned and passed to the model (Dataset 1)
CATEGORICAL_FEATURES = [
    "ed_site", "gender", "ethnicity", "race", "language", "marital_status",
    "employment_status", "insurance_status", "arrival_mode", "arrival_month",
    "arrival_day", "arrival_hour_bin", "chief_complaint", "previous_disposition",
]

NUMERIC_FEATURES = [
    "esi_level", "age", "prior_ed_visits", "prior_admissions", "prior_surgeries",
    "prior_imaging_tests", "prior_ekgs", "num_comorbidities",
    "hypertension", "diabetes", "heart_failure", "coronary_disease", "arrhythmia",
    "copd_asthma", "kidney_disease", "cancer_history", "mental_health_disorder",
    "substance_use_disorder", "heart_rate", "systolic_bp", "diastolic_bp",
    "respiratory_rate", "oxygen_saturation", "on_supplemental_oxygen", "temperature_c",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Present in the data but NOT used by the final model: the home-medication list is
# usually completed (medication reconciliation) during the admission itself, so it
# leaks the outcome. Section 7b of the main script measures this leakage.
LEAKAGE_SUSPECTS = ["home_medications"]

# Physiologically possible ranges; values outside are treated as data-entry errors
VITAL_LIMITS = {
    "heart_rate": (20, 250),
    "systolic_bp": (50, 300),
    "diastolic_bp": (20, 200),
    "respiratory_rate": (5, 60),
    "oxygen_saturation": (50, 100),
    "temperature_c": (32, 43),
}

# Placeholder answers that really mean "missing"
UNKNOWN_VALUES = ["Unknown", "Patient Refused", "unknown", "", " "]

N_TOP_COMPLAINTS = 30

# Colors (validated categorical palette, fixed order)
COLOR_DISCHARGE = "#2a78d6"
COLOR_ADMIT = "#eb6834"
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                "#6250d6", "#e34948"]
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def clean_patient_flow(df, top_complaints=None):
    """Clean Dataset 1 (ED patient flow). Returns the cleaned copy and the list
    of chief complaints kept as their own category."""
    df = df.copy()

    # Text columns: trim spaces, unify spelling ('ambulance' vs 'Car'), unknown -> NaN
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace(UNKNOWN_VALUES, pd.NA)
    df["arrival_mode"] = df["arrival_mode"].str.title()
    df["ed_site"] = "Site " + df["ed_site"].astype("string").str.replace("Site ", "")

    # Temperature: Fahrenheit -> Celsius (KTAS dataset and most of the world use C)
    if "temperature_f" in df.columns:
        df["temperature_c"] = ((df["temperature_f"] - 32) * 5 / 9).round(1)
        df = df.drop(columns="temperature_f")

    # Impossible vital signs -> missing (they are later imputed)
    for col, (low, high) in VITAL_LIMITS.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

    # ESI must be 1-5
    df["esi_level"] = pd.to_numeric(df["esi_level"], errors="coerce")
    df.loc[~df["esi_level"].isin([1, 2, 3, 4, 5]), "esi_level"] = np.nan

    # Rare chief complaints are grouped into 'other'
    if top_complaints is None:
        top_complaints = (df["chief_complaint"].value_counts()
                          .drop("other", errors="ignore")
                          .head(N_TOP_COMPLAINTS).index.tolist())
    df["chief_complaint"] = df["chief_complaint"].where(
        df["chief_complaint"].isin(top_complaints) | df["chief_complaint"].isna(), "other")

    if "disposition" in df.columns:
        df[TARGET] = (df["disposition"] == "Admit").astype(int)

    # pandas 'string' dtype -> plain object so scikit-learn handles NaN normally
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

    return df, top_complaints


KTAS_ADMIT_CODES = [2, 3, 7]      # ward, ICU, operating room from ED
KTAS_NOT_ADMIT_CODES = [1, 4]     # discharge, discharge against medical advice
KTAS_DISPOSITION_NAMES = {1: "Discharge", 2: "Ward admission", 3: "ICU admission",
                          4: "AMA discharge", 5: "Transfer", 6: "Death",
                          7: "Operation from ED"}
KTAS_ARRIVAL_NAMES = {1: "Walking", 2: "Public ambulance (119)", 3: "Private car",
                      4: "Private ambulance", 5: "Public transport", 6: "Wheelchair",
                      7: "Others"}


def clean_ktas(df):
    """Clean Dataset 2 (Korean KTAS emergency department records)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Vital signs contain the Korean word '측불' (= "unmeasurable") -> NaN
    for col in ["SBP", "DBP", "HR", "RR", "BT", "Saturation"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    limits = {"SBP": (50, 300), "DBP": (20, 200), "HR": (20, 250), "RR": (5, 60),
              "BT": (32, 43), "Saturation": (50, 100)}
    for col, (low, high) in limits.items():
        df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

    # Pain score is only recorded when the patient has pain; no pain -> score 0
    df.loc[df["Pain"] == 2, "NRS_pain"] = df.loc[df["Pain"] == 2, "NRS_pain"].fillna(0)

    # Transfers and deaths are neither 'admitted here' nor 'sent home' -> removed
    df = df[df["Disposition"].isin(KTAS_ADMIT_CODES + KTAS_NOT_ADMIT_CODES)].copy()
    df[TARGET] = df["Disposition"].isin(KTAS_ADMIT_CODES).astype(int)
    return df


def harmonise_patient_flow(df):
    """Common triage features shared by both datasets (for cross-dataset testing)."""
    return pd.DataFrame({
        "age": df["age"],
        "male": (df["gender"] == "Male").astype(float),
        "acuity_level": df["esi_level"],
        "arrived_by_ambulance": (df["arrival_mode"] == "Ambulance").astype(float),
        "heart_rate": df["heart_rate"],
        "systolic_bp": df["systolic_bp"],
        "diastolic_bp": df["diastolic_bp"],
        "respiratory_rate": df["respiratory_rate"],
        "temperature_c": df["temperature_c"],
        "oxygen_saturation": df["oxygen_saturation"],
    })


def harmonise_ktas(df):
    return pd.DataFrame({
        "age": df["Age"],
        "male": (df["Sex"] == 2).astype(float),
        "acuity_level": df["KTAS_RN"],
        "arrived_by_ambulance": df["Arrival mode"].isin([2, 4]).astype(float),
        "heart_rate": df["HR"],
        "systolic_bp": df["SBP"],
        "diastolic_bp": df["DBP"],
        "respiratory_rate": df["RR"],
        "temperature_c": df["BT"],
        "oxygen_saturation": df["Saturation"],
    })


def calibrate_probability(score, prior_ratio):
    """The model is trained with class_weight='balanced', which multiplies the odds
    of admission by (n_discharged / n_admitted). Dividing the odds by the same
    ratio gives back a realistic probability (used for bed-demand forecasting)."""
    score = np.clip(score, 1e-6, 1 - 1e-6)
    odds = score / (1 - score) / prior_ratio
    return odds / (1 + odds)


def risk_band(score, threshold):
    """Translate the model score into an action for the bed manager."""
    if score >= threshold:
        return "HIGH", "Admission predicted - alert bed management and start the admission work-up early."
    if score >= 0.40:
        return "MODERATE", "Borderline case - reassess after first tests and keep a bed on standby."
    return "LOW", "Discharge predicted - suitable for the fast-track / treat-and-release pathway."
