"""
CT-264 Programming for AI - CCA Project
AI-Based Patient Admission Prediction System

Step 1: Prepare the two datasets used in this project.

Dataset 1 - Emergency Department Patient Flow (Yale New Haven Health, 2014-2017)
    Hong WS, Haimovich AD, Taylor RA. "Predicting hospital admission at emergency
    department triage using machine learning." PLOS ONE 13(7): e0201016 (2018).
    Source: https://zenodo.org/records/1308993  (also mirrored on Kaggle as
    "Hospital Triage and Patient History Data").
    Original file: 5v_cleandf.RData -> 560,486 ED visits x 972 columns.

Dataset 2 - Korean Triage and Acuity Scale (KTAS) ED dataset (2016-2017)
    Moon SH, Shim JL, Park KS, Park CS. "Triage accuracy and causes of mistriage
    using the Korean Triage and Acuity Scale." PLOS ONE 14(9): e0216972 (2019).
    Source: PLOS ONE supporting information S1 (also on Kaggle as
    "Emergency Service - Triage Application").
    Original file: S1 Data (xlsx) -> 1,267 ED visits x 24 columns.

This script only needs to be run once. The ready-made CSV files are already in the
data/ folder, so the main project (02_admission_prediction.py) runs without it.
Running it again needs the raw downloads in data/raw_download/ and the
"pyreadr" package (pip install pyreadr).
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_download")
DATA_DIR = os.path.join(BASE_DIR, "data")

YALE_RDATA = os.path.join(RAW_DIR, "yale", "yaleemmlc-admissionprediction-5cf50bc",
                          "Results", "5v_cleandf.RData")
KTAS_XLSX = os.path.join(RAW_DIR, "ktas_s1.xlsx")

SAMPLE_SIZE = 100_000      # visits kept from the 560,486 so the project runs on a laptop
RANDOM_STATE = 42


def prepare_yale_dataset():
    import pyreadr

    print("Reading the Yale RData file (this takes about a minute)...")
    df = pyreadr.read_r(YALE_RDATA)["df"]
    print("Original shape:", df.shape)

    cols = list(df.columns)

    # Past medical history = the AHRQ CCS diagnosis columns between
    # 'previousdispo' and 'n_edvisits'
    pmh_cols = cols[cols.index("previousdispo") + 1: cols.index("n_edvisits")]
    med_cols = [c for c in cols if c.startswith("meds_")]
    cc_cols = [c for c in cols if c.startswith("cc_")]
    img_cols = ["cxr_count", "echo_count", "headct_count", "mri_count", "otherct_count",
                "otherimg_count", "otherus_count", "otherxr_count"]

    # Chief complaint: one text column instead of 200 one-hot columns.
    # When a visit has more than one complaint, the most common one is kept.
    cc = df[cc_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    order = cc.sum().sort_values(ascending=False).index
    cc = cc[order]
    has_cc = cc.sum(axis=1) > 0
    chief = cc.idxmax(axis=1).str.replace("cc_", "", regex=False)
    chief[~has_cc] = np.nan

    pmh = df[pmh_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    def flag(*names):
        return (pmh[list(names)].sum(axis=1) > 0).astype(int)

    out = pd.DataFrame({
        "ed_site": df["dep_name"].astype(str),
        "esi_level": pd.to_numeric(df["esi"].astype(str), errors="coerce"),
        "age": pd.to_numeric(df["age"].astype(str), errors="coerce"),
        "gender": df["gender"].astype(str),
        "ethnicity": df["ethnicity"].astype(str),
        "race": df["race"].astype(str),
        "language": df["lang"].astype(str),
        "marital_status": df["maritalstatus"].astype(str),
        "employment_status": df["employstatus"].astype(str),
        "insurance_status": df["insurance_status"].astype(str),
        "arrival_mode": df["arrivalmode"].astype(object),
        "arrival_month": df["arrivalmonth"].astype(str),
        "arrival_day": df["arrivalday"].astype(str),
        "arrival_hour_bin": df["arrivalhour_bin"].astype(str),
        "chief_complaint": chief,
        "previous_disposition": df["previousdispo"].astype(str),
        "prior_ed_visits": df["n_edvisits"].astype(float),
        "prior_admissions": df["n_admissions"].astype(float),
        "prior_surgeries": df["n_surgeries"].astype(float),
        "prior_imaging_tests": df[img_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1),
        "prior_ekgs": pd.to_numeric(df["ekg_count"], errors="coerce"),
        "home_medications": df[med_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1),
        "num_comorbidities": (pmh > 0).sum(axis=1),
        "hypertension": flag("htn", "htncomplicn"),
        "diabetes": flag("diabmelnoc", "diabmelwcm"),
        "heart_failure": flag("chfnonhp"),
        "coronary_disease": flag("coronathero", "acutemi"),
        "arrhythmia": flag("dysrhythmia"),
        "copd_asthma": flag("copd", "asthma"),
        "kidney_disease": flag("chrkidneydisease", "acrenlfail"),
        "cancer_history": flag("maligneopls", "brnchlngca", "breastcancr", "coloncancer",
                               "prostatecan", "leukemias", "nonhodglym", "otprimryca"),
        "mental_health_disorder": flag("mooddisorders", "anxietydisorders",
                                       "schizophreniaandotherpsychoticdisorde"),
        "substance_use_disorder": flag("substancerelateddisorders", "alcoholrelateddisorders"),
        "heart_rate": df["triage_vital_hr"],
        "systolic_bp": df["triage_vital_sbp"],
        "diastolic_bp": df["triage_vital_dbp"],
        "respiratory_rate": df["triage_vital_rr"],
        "oxygen_saturation": df["triage_vital_o2"],
        "on_supplemental_oxygen": df["triage_vital_o2_device"],
        "temperature_f": df["triage_vital_temp"],
        "disposition": df["disposition"].astype(str),
    })

    # 'nan' strings created by astype(str) are turned back into real missing values
    out = out.replace({"nan": np.nan})

    sample = out.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    sample.insert(0, "visit_id", np.arange(1, len(sample) + 1))

    path = os.path.join(DATA_DIR, "ed_patient_flow.csv")
    sample.to_csv(path, index=False)
    print(f"Saved {path}  shape={sample.shape}")
    print(sample["disposition"].value_counts(normalize=True).round(3))


def prepare_ktas_dataset():
    df = pd.read_excel(KTAS_XLSX)
    path = os.path.join(DATA_DIR, "ktas_triage.csv")
    # utf-8-sig keeps the Korean text readable when the CSV is opened in Excel
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved {path}  shape={df.shape}")


if __name__ == "__main__":
    prepare_ktas_dataset()
    prepare_yale_dataset()
