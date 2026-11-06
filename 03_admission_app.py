"""
CT-264 Programming for AI - CCA Project
AI-Based Patient Admission Prediction System - User Interface

A simple desktop app (Tkinter, comes with Python) for ED staff / bed managers:
  Tab 1  Predict Admission : enter triage details -> admission risk + action
  Tab 2  Model Performance : comparison tables and all project charts
  Tab 3  About             : project, datasets and group members

Run 02_admission_prediction.py first (it creates outputs/models/...), then:
    python 03_admission_app.py
    python 03_admission_app.py --test     (quick check without opening a window)
    python 03_admission_app.py --screenshot report   (saves screenshots for the report)
"""

import os
import sys
import json
import random

import joblib
import numpy as np
import pandas as pd

from admission_utils import (FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                             clean_patient_flow, calibrate_probability, risk_band)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
METRIC_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")

BAND_COLORS = {"HIGH": "#c0392b", "MODERATE": "#d68910", "LOW": "#1e8449"}

CONDITIONS = [("hypertension", "Hypertension"), ("diabetes", "Diabetes"),
              ("heart_failure", "Heart failure"), ("coronary_disease", "Coronary disease"),
              ("arrhythmia", "Arrhythmia"), ("copd_asthma", "COPD / asthma"),
              ("kidney_disease", "Kidney disease"), ("cancer_history", "Cancer history"),
              ("mental_health_disorder", "Mental health disorder"),
              ("substance_use_disorder", "Substance use disorder")]

VITALS = [("heart_rate", "Heart rate (bpm)"), ("systolic_bp", "Systolic BP (mmHg)"),
          ("diastolic_bp", "Diastolic BP (mmHg)"), ("respiratory_rate", "Resp. rate (/min)"),
          ("oxygen_saturation", "SpO2 (%)"), ("temperature_c", "Temperature (°C)")]

HISTORY = [("prior_ed_visits", "ED visits (last year)"),
           ("prior_admissions", "Admissions (last year)"),
           ("prior_surgeries", "Previous surgeries"),
           ("prior_imaging_tests", "Previous imaging tests"),
           ("prior_ekgs", "Previous ECGs"),
           ("num_comorbidities", "No. of past conditions")]


class AdmissionPredictor:
    """Loads the saved model and turns one patient record into a prediction."""

    def __init__(self):
        model_path = os.path.join(MODEL_DIR, "admission_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Run 02_admission_prediction.py first.")
        self.model = joblib.load(model_path)
        with open(os.path.join(MODEL_DIR, "model_info.json")) as f:
            self.info = json.load(f)
        self.threshold = self.info["threshold"]
        self.demo = pd.read_csv(os.path.join(MODEL_DIR, "demo_patients.csv"))

    def predict(self, patient):
        row = pd.DataFrame([{c: patient.get(c, np.nan) for c in FEATURES}])
        row, _ = clean_patient_flow(row, self.info["top_complaints"])
        score = float(self.model.predict_proba(row[FEATURES])[:, 1][0])
        probability = float(calibrate_probability(score, self.info["prior_ratio"]))
        band, action = risk_band(score, self.threshold)
        return {"score": score, "probability": probability, "band": band, "action": action,
                "admit": score >= self.threshold}


def run_self_test():
    p = AdmissionPredictor()
    print(f"Model: {p.info['model_name']}  (threshold {p.threshold:.2f})")
    correct = 0
    sample = p.demo.head(200)
    for _, r in sample.iterrows():
        res = p.predict(r.to_dict())
        correct += int(res["admit"] == bool(r["actual_admitted"]))
    print(f"Accuracy on 200 demo patients: {correct / len(sample):.1%}")
    for _, r in p.demo.head(5).iterrows():
        res = p.predict(r.to_dict())
        print(f"  age {r['age']:>4.0f}, ESI {r['esi_level']}, {r['chief_complaint']:<22} -> "
              f"{res['band']:<8} score {res['score']:.2f}  prob {res['probability']:.0%}  "
              f"actual: {'Admitted' if r['actual_admitted'] else 'Discharged'}")


def launch_gui(screenshot_dir=None):
    import tkinter as tk
    from tkinter import ttk, messagebox
    try:   # sharp text on high-resolution Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    predictor = AdmissionPredictor()
    info = predictor.info
    cats = info["categories"]
    defaults = info["defaults"]

    root = tk.Tk()
    root.title("AI-Based Patient Admission Prediction System")
    root.geometry("1180x800")
    root.minsize(1050, 720)

    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#1c5cab")
    style.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#52514e")
    style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=6)

    ttk.Label(root, text="AI-Based Patient Admission Prediction System",
              style="Header.TLabel").pack(anchor="w", padx=14, pady=(10, 0))
    ttk.Label(root, text=f"Emergency department triage decision support  |  Model: "
                         f"{info['model_name']}  |  Test ROC-AUC {info['test_metrics']['ROC-AUC']:.3f}",
              style="Sub.TLabel").pack(anchor="w", padx=14)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=8)

    # ------------------------------------------------------------ Tab 1
    tab = ttk.Frame(notebook, padding=8)
    notebook.add(tab, text="  Predict Admission  ")
    form = ttk.Frame(tab)
    form.pack(side="left", fill="both", expand=True)
    result = ttk.LabelFrame(tab, text=" Prediction ", padding=12)
    result.pack(side="right", fill="y", padx=(10, 0))

    widgets = {}

    def add_combo(parent, key, label, r, values, width=22):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=str(defaults.get(key, values[0])))
        cb = ttk.Combobox(parent, textvariable=var, values=values, width=width, state="readonly")
        cb.grid(row=r, column=1, sticky="w", pady=2, padx=4)
        widgets[key] = var

    def add_entry(parent, key, label, r, value=""):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=value)
        ttk.Entry(parent, textvariable=var, width=12).grid(row=r, column=1, sticky="w", pady=2, padx=4)
        widgets[key] = var

    col1 = ttk.Frame(form)
    col1.grid(row=0, column=0, sticky="nsew", padx=4)
    col2 = ttk.Frame(form)
    col2.grid(row=0, column=1, sticky="nsew", padx=4)
    col3 = ttk.Frame(form)
    col3.grid(row=0, column=2, sticky="nsew", padx=4)

    f_patient = ttk.LabelFrame(col1, text=" Patient ", padding=8)
    f_patient.pack(fill="x", pady=4)
    add_entry(f_patient, "age", "Age (years)", 0, "45")
    add_combo(f_patient, "gender", "Gender", 1, cats["gender"])
    add_combo(f_patient, "insurance_status", "Insurance", 2, cats["insurance_status"])
    add_combo(f_patient, "marital_status", "Marital status", 3, cats["marital_status"])
    add_combo(f_patient, "employment_status", "Employment", 4, cats["employment_status"])
    add_combo(f_patient, "race", "Race", 5, cats["race"])
    add_combo(f_patient, "ethnicity", "Ethnicity", 6, cats["ethnicity"])
    add_combo(f_patient, "language", "Language", 7, cats["language"])

    f_arrival = ttk.LabelFrame(col1, text=" Arrival & triage ", padding=8)
    f_arrival.pack(fill="x", pady=4)
    add_combo(f_arrival, "esi_level", "Triage level (ESI)", 0, ["1", "2", "3", "4", "5"], width=6)
    widgets["esi_level"].set("3")
    add_combo(f_arrival, "chief_complaint", "Chief complaint", 1, cats["chief_complaint"])
    add_combo(f_arrival, "arrival_mode", "Arrival mode", 2, cats["arrival_mode"])
    add_combo(f_arrival, "ed_site", "ED site", 3, cats["ed_site"])
    add_combo(f_arrival, "arrival_day", "Arrival day", 4, cats["arrival_day"])
    add_combo(f_arrival, "arrival_hour_bin", "Arrival time", 5, cats["arrival_hour_bin"])
    add_combo(f_arrival, "arrival_month", "Arrival month", 6, cats["arrival_month"])

    f_vitals = ttk.LabelFrame(col2, text=" Triage vital signs (blank = not measured) ", padding=8)
    f_vitals.pack(fill="x", pady=4)
    for i, (key, label) in enumerate(VITALS):
        add_entry(f_vitals, key, label, i, f"{defaults[key]:g}")
    o2 = tk.IntVar(value=0)
    ttk.Checkbutton(f_vitals, text="On supplemental oxygen", variable=o2).grid(
        row=len(VITALS), column=0, columnspan=2, sticky="w", pady=2)
    widgets["on_supplemental_oxygen"] = o2

    f_hist = ttk.LabelFrame(col2, text=" Hospital history ", padding=8)
    f_hist.pack(fill="x", pady=4)
    for i, (key, label) in enumerate(HISTORY):
        add_entry(f_hist, key, label, i, "0")
    add_combo(f_hist, "previous_disposition", "Previous ED outcome", len(HISTORY),
              cats["previous_disposition"], width=18)

    f_cond = ttk.LabelFrame(col3, text=" Known conditions ", padding=8)
    f_cond.pack(fill="x", pady=4)
    for i, (key, label) in enumerate(CONDITIONS):
        v = tk.IntVar(value=0)
        ttk.Checkbutton(f_cond, text=label, variable=v).grid(row=i, column=0, sticky="w", pady=1)
        widgets[key] = v

    def read_form():
        patient = {}
        for key, var in widgets.items():
            value = var.get()
            if key in NUMERIC_FEATURES:
                value = str(value).strip()
                if value == "":
                    patient[key] = np.nan
                    continue
                try:
                    patient[key] = float(value)
                except ValueError:
                    raise ValueError(f"'{value}' is not a number ({key.replace('_', ' ')})")
            else:
                patient[key] = value
        if not (0 <= patient.get("age", 0) <= 120):
            raise ValueError("Age must be between 0 and 120.")
        return patient

    def fill_form(record):
        for key, var in widgets.items():
            value = record.get(key, "")
            if isinstance(var, tk.IntVar):
                var.set(int(value) if pd.notna(value) else 0)
            elif key in NUMERIC_FEATURES:
                var.set("" if pd.isna(value) else f"{float(value):g}")
            else:
                var.set("" if pd.isna(value) else str(value))
        widgets["esi_level"].set("" if pd.isna(record.get("esi_level")) else str(int(record["esi_level"])))

    # result panel
    band_lbl = tk.Label(result, text="—", font=("Segoe UI", 26, "bold"), width=12, fg="white", bg="#a3a29d")
    band_lbl.pack(pady=(4, 8))
    decision_lbl = ttk.Label(result, text="Enter patient details and press Predict",
                             font=("Segoe UI", 11, "bold"), wraplength=260)
    decision_lbl.pack(pady=2)
    ttk.Label(result, text="Model risk score").pack(anchor="w", pady=(10, 0))
    score_bar = ttk.Progressbar(result, length=260, maximum=100)
    score_bar.pack()
    score_lbl = ttk.Label(result, text="")
    score_lbl.pack(anchor="w")
    prob_lbl = ttk.Label(result, text="", font=("Segoe UI", 10))
    prob_lbl.pack(anchor="w", pady=(8, 0))
    action_lbl = ttk.Label(result, text="", wraplength=260, foreground="#333333")
    action_lbl.pack(anchor="w", pady=8)
    actual_lbl = ttk.Label(result, text="", wraplength=260, font=("Segoe UI", 10, "italic"))
    actual_lbl.pack(anchor="w", pady=4)
    ttk.Separator(result).pack(fill="x", pady=8)
    ttk.Label(result, text=f"Decision threshold: {predictor.threshold:.2f}\n"
                           "Score ≥ threshold → admission predicted.\n"
                           "Probability is corrected for the class\n"
                           "weighting used in training.", style="Sub.TLabel").pack(anchor="w")

    def show(res, actual=None):
        band_lbl.config(text=res["band"], bg=BAND_COLORS[res["band"]])
        decision_lbl.config(text="ADMISSION PREDICTED" if res["admit"] else "DISCHARGE PREDICTED")
        score_bar["value"] = res["score"] * 100
        score_lbl.config(text=f"{res['score']:.2f}  (0 = discharge, 1 = admit)")
        prob_lbl.config(text=f"Estimated chance of admission: {res['probability']:.0%}")
        action_lbl.config(text="Recommended action: " + res["action"])
        if actual is None:
            actual_lbl.config(text="")
        else:
            ok = bool(actual) == res["admit"]
            actual_lbl.config(text=f"Real outcome of this test patient: "
                                   f"{'ADMITTED' if actual else 'DISCHARGED'}  "
                                   f"({'model correct' if ok else 'model wrong'})")

    current_actual = {"value": None}

    def on_predict():
        try:
            res = predictor.predict(read_form())
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        show(res, current_actual["value"])

    def on_demo():
        rec = predictor.demo.iloc[random.randrange(len(predictor.demo))].to_dict()
        fill_form(rec)
        current_actual["value"] = int(rec["actual_admitted"])
        show(predictor.predict(rec), current_actual["value"])

    def on_reset():
        current_actual["value"] = None
        for key, var in widgets.items():
            if isinstance(var, tk.IntVar):
                var.set(0)
            elif key in [k for k, _ in VITALS]:
                var.set(f"{defaults[key]:g}")
            elif key in NUMERIC_FEATURES:
                var.set("0")
            else:
                var.set(str(defaults.get(key, "")))
        widgets["age"].set("45")
        widgets["esi_level"].set("3")
        band_lbl.config(text="—", bg="#a3a29d")
        decision_lbl.config(text="Enter patient details and press Predict")
        for lbl in (score_lbl, prob_lbl, action_lbl, actual_lbl):
            lbl.config(text="")
        score_bar["value"] = 0

    def user_edit(*_):
        current_actual["value"] = None

    buttons = ttk.Frame(col3)
    buttons.pack(fill="x", pady=12)
    ttk.Button(buttons, text="Predict", style="Big.TButton", command=on_predict).pack(fill="x", pady=3)
    ttk.Button(buttons, text="Load random test patient", command=on_demo).pack(fill="x", pady=3)
    ttk.Button(buttons, text="Reset form", command=on_reset).pack(fill="x", pady=3)
    ttk.Label(col3, text="Tip: 'Load random test patient' fills\nthe form with a real visit from the\n"
                         "unseen test set and shows its true\noutcome.", style="Sub.TLabel").pack(anchor="w")
    for key in ("age", "chief_complaint", "esi_level"):
        widgets[key].trace_add("write", user_edit)

    # ------------------------------------------------------------ Tab 2
    perf = ttk.Frame(notebook, padding=8)
    notebook.add(perf, text="  Model Performance  ")

    def add_table(parent, csv_name, title, height):
        df = pd.read_csv(os.path.join(METRIC_DIR, csv_name))
        box = ttk.LabelFrame(parent, text=f" {title} ", padding=4)
        box.pack(fill="x", pady=4)
        tree = ttk.Treeview(box, columns=list(df.columns), show="headings", height=height)
        for c in df.columns:
            tree.heading(c, text=c)
            tree.column(c, width=170 if c == df.columns[0] else 95, anchor="w" if c == df.columns[0] else "center")
        for _, r in df.iterrows():
            tree.insert("", "end", values=[f"{v:.3f}" if isinstance(v, float) else v for v in r])
        tree.pack(fill="x")

    top = ttk.Frame(perf)
    top.pack(fill="x")
    add_table(top, "model_comparison.csv", "Dataset 1 (USA, 20,000 test visits)", 7)
    add_table(top, "ktas_model_comparison.csv", "Dataset 2 (Korea KTAS, 5-fold cross-validation)", 4)

    viewer = ttk.Frame(perf)
    viewer.pack(fill="both", expand=True, pady=4)
    figs = sorted(f for f in os.listdir(FIG_DIR) if f.endswith(".png"))
    fig_var = tk.StringVar(value=figs[11] if len(figs) > 11 else figs[0])
    bar = ttk.Frame(viewer)
    bar.pack(fill="x")
    ttk.Label(bar, text="Chart: ").pack(side="left")
    ttk.Combobox(bar, textvariable=fig_var, values=figs, width=45, state="readonly").pack(side="left")
    img_lbl = ttk.Label(viewer)
    img_lbl.pack(pady=4)

    def show_fig(*_):
        img = tk.PhotoImage(file=os.path.join(FIG_DIR, fig_var.get()))
        factor = max(1, int(np.ceil(max(img.width() / 1100, img.height() / 330))))
        img = img.subsample(factor, factor)
        img_lbl.configure(image=img)
        img_lbl.image = img

    fig_var.trace_add("write", show_fig)
    show_fig()

    # ------------------------------------------------------------ Tab 3
    about = ttk.Frame(notebook, padding=16)
    notebook.add(about, text="  About  ")
    text = (
        "CT-264 Programming for Artificial Intelligence - Complex Computing Activity\n"
        "NED University of Engineering and Technology, Department of CSIT\n\n"
        "Group members\n"
        "   CT-24033   Muhammad Hamza Siddiqui\n"
        "   CT-24044   Saif Ur Rehman\n"
        "   CT-24312   Muhammad Talha Bin Suhail\n\n"
        "Purpose\n"
        "   Predict at triage whether an emergency patient will be admitted, so beds and\n"
        "   staff can be prepared earlier and ED crowding is reduced.\n\n"
        "Datasets\n"
        "   1. ED Patient Flow - Yale New Haven Health (Hong et al., PLOS ONE 2018),\n"
        "      100,000 visits sampled from 560,486 visits.\n"
        "   2. Korean Triage and Acuity Scale dataset (Moon et al., PLOS ONE 2019), 1,267 visits.\n\n"
        f"Final model: {info['model_name']} trained on {info['train_size']:,} visits,\n"
        f"tested on {info['test_size']:,} unseen visits (ROC-AUC {info['test_metrics']['ROC-AUC']:.3f}, "
        f"recall {info['test_metrics']['Recall (sensitivity)']:.3f}).\n\n"
        "Disclaimer: decision-support prototype for education only. It must not replace\n"
        "clinical judgement and would need local validation before real use."
    )
    ttk.Label(about, text=text, font=("Segoe UI", 10), justify="left").pack(anchor="w")

    if screenshot_dir:
        def capture(name):
            from PIL import ImageGrab
            root.update()
            x, y = root.winfo_rootx(), root.winfo_rooty()
            w, h = root.winfo_width(), root.winfo_height()
            ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(os.path.join(screenshot_dir, name))

        def run_capture():
            random.seed(11)
            on_demo()
            root.after(400, lambda: (capture("gui_predict.png"), notebook.select(1),
                                     root.after(600, lambda: (capture("gui_performance.png"),
                                                              notebook.select(2),
                                                              root.after(400, lambda: (capture("gui_about.png"),
                                                                                       root.destroy()))))))
        os.makedirs(screenshot_dir, exist_ok=True)
        root.attributes("-topmost", True)
        root.after(1500, run_capture)

    root.mainloop()


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_self_test()
    elif "--screenshot" in sys.argv:
        launch_gui(screenshot_dir=sys.argv[sys.argv.index("--screenshot") + 1])
    else:
        launch_gui()
