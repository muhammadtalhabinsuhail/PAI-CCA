# AI-Based Patient Admission Prediction System

CT-264 Programming for Artificial Intelligence - Complex Computing Activity (CCA)
NED University of Engineering and Technology, Department of CSIT

| Roll No. | Name |
|---|---|
| CT-24033 | Muhammad Hamza Siddiqui |
| CT-24044 | Saif Ur Rehman |
| CT-24312 | Muhammad Talha Bin Suhail |

Predicts at emergency department (ED) triage whether a patient will be **admitted** or
**discharged**, to help hospitals plan beds and staff earlier.

## How to run

```
pip install -r requirements.txt
python 02_admission_prediction.py     # about 5-6 minutes: preprocessing, EDA, models, evaluation
python 03_admission_app.py            # opens the prediction interface (GUI)
python 03_admission_app.py --test     # quick check of the saved model without a window
```

`01_prepare_datasets.py` does not need to be run: the two CSV files it creates are already in `data/`.

Note: if your scikit-learn version differs from the one used to save the model, run
`02_admission_prediction.py` once before the app so the model is saved again with your version.

## Files

| File / folder | Purpose |
|---|---|
| `01_prepare_datasets.py` | Builds the CSV datasets from the raw downloads (run once) |
| `02_admission_prediction.py` | Main project: data quality, preprocessing, EDA, class imbalance, 4 models, tuning, ensemble, evaluation, leakage check, Dataset 2, external validation, bed-demand forecast |
| `03_admission_app.py` | Tkinter interface: prediction form, risk level, recommended action, model-performance dashboard |
| `admission_utils.py` | Shared cleaning functions used by the scripts and the app |
| `data/ed_patient_flow.csv` | Dataset 1: 100,000 ED visits (Yale New Haven Health, USA) |
| `data/ktas_triage.csv` | Dataset 2: 1,267 ED visits (Korean Triage and Acuity Scale) |
| `outputs/figures/` | 19 charts |
| `outputs/metrics/` | Result tables (CSV) |
| `outputs/models/` | Saved model, model information, demo patients |
| `outputs/console_output.txt` | Full console output of the main script |
| `report/` | Technical report (Word + PDF) |

## Datasets

1. Hong WS, Haimovich AD, Taylor RA. *Predicting hospital admission at emergency department triage
   using machine learning.* PLOS ONE 13(7): e0201016 (2018).
   Data: https://github.com/yaleemmlc/admissionprediction (Zenodo record 1308993), also on Kaggle.
2. Moon SH, Shim JL, Park KS, Park CS. *Triage accuracy and causes of mistriage using the Korean
   Triage and Acuity Scale.* PLOS ONE 14(9): e0216972 (2019), S1 Data, also on Kaggle
   ("Emergency Service - Triage Application").

## Main results (test set, 20,000 unseen visits)

Final model: Tuned Gradient Boosting - ROC-AUC 0.885, recall 0.829, F1 0.703, accuracy 0.792.
Korean dataset (5-fold CV): Random Forest ROC-AUC 0.808. External validation USA to Korea: ROC-AUC 0.760.
