"""
================================================================================
  Hospital Patient Flow and Operational Efficiency Prediction Using ML
  Author  : Sourajit Mondal
  Program : IBM Bharat / AICTE Data Analytics with AI Virtual Internship
  File    : SourajitMondal_HospitalPatientFlow.py
================================================================================

USAGE
-----
  Run Flask backend  :  python SourajitMondal_HospitalPatientFlow.py flask
  Run Streamlit UI   :  streamlit run SourajitMondal_HospitalPatientFlow.py
  Run training only  :  python SourajitMondal_HospitalPatientFlow.py train
  Default (no args)  :  python SourajitMondal_HospitalPatientFlow.py
                        -> trains model and starts Flask

DATASET COLUMNS (11 total)
---------------------------
  Patient Id, Patient Admission Date, Patient Admission Time, Merged,
  Patient Gender, Patient Age, Patient Race, Department Referral,
  Patient Admission Flag, Patient Satisfaction Score, Patient Waittime

TARGET  : Patient Waittime  (regression)
REASON  : Waiting time is the primary operational KPI for patient flow
          management. It directly impacts patient satisfaction and resource
          planning.
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import warnings
import io
import base64
import threading
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(BASE_DIR, "healthcare_analytics_patient_flow_data.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "trained_model.joblib")
FLASK_PORT  = 5050

TARGET_COL   = "Patient Waittime"
CAT_FEATURES = ["Patient Gender", "Patient Race", "Department Referral",
                "Patient Admission Flag"]
NUM_FEATURES = ["Patient Age"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# Chart colour palette — purple-primary health-tech theme
C_BLUE   = "#7C3AED"   # primary purple
C_PURPLE = "#8B5CF6"   # lighter purple
C_GREEN  = "#10B981"   # success green
C_ORANGE = "#F59E0B"   # warning amber
C_RED    = "#EF4444"   # danger red
C_TEAL   = "#06B6D4"   # cyan/teal accent
BG_CHART = "#FFFFFF"
GRID_CLR = "#F3F0FF"   # very light purple grid


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – DATA LOADING & PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def load_raw_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load CSV and return raw DataFrame."""
    df = pd.read_csv(csv_path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the raw DataFrame."""
    df = df.copy()

    # Strip whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Parse dates/times
    df["Patient Admission Date"] = pd.to_datetime(
        df["Patient Admission Date"], errors="coerce"
    )
    df["Admission Hour"] = pd.to_datetime(
        df["Patient Admission Time"], format="%I:%M:%S %p", errors="coerce"
    ).dt.hour
    df["Admission DayOfWeek"] = df["Patient Admission Date"].dt.dayofweek
    df["Admission Month"]     = df["Patient Admission Date"].dt.month

    # Fill missing numeric values with median
    df["Patient Satisfaction Score"] = df["Patient Satisfaction Score"].fillna(
        df["Patient Satisfaction Score"].median()
    )
    df["Patient Waittime"] = df["Patient Waittime"].fillna(
        df["Patient Waittime"].median()
    )

    # Standardise department referral
    df["Department Referral"] = df["Department Referral"].replace(
        {"None": "No Referral", None: "No Referral"}
    ).fillna("No Referral")

    # Age Group
    bins   = [0, 12, 17, 35, 60, 120]
    labels = ["Child (0-12)", "Teen (13-17)", "Young Adult (18-35)",
              "Adult (36-60)", "Senior (60+)"]
    df["Age Group"] = pd.cut(
        df["Patient Age"], bins=bins, labels=labels, right=True
    ).astype(str)

    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a JSON-serialisable summary of the processed dataset."""
    return {
        "total_records"        : int(len(df)),
        "total_columns"        : int(len(df.columns)),
        "missing_values"       : int(df.isnull().sum().sum()),
        "duplicate_records"    : int(df.duplicated().sum()),
        "avg_wait_time_min"    : round(float(df["Patient Waittime"].mean()), 2),
        "median_wait_time_min" : round(float(df["Patient Waittime"].median()), 2),
        "avg_patient_age"      : round(float(df["Patient Age"].mean()), 1),
        "avg_satisfaction"     : round(float(df["Patient Satisfaction Score"].mean()), 2),
        "admission_rate_pct"   : round(
            float((df["Patient Admission Flag"] == "Admission").mean() * 100), 1
        ),
        "departments"          : df["Department Referral"].nunique(),
        "gender_distribution"  : df["Patient Gender"].value_counts().to_dict(),
        "top_departments"      : df["Department Referral"].value_counts().head(6).to_dict(),
        "race_distribution"    : df["Patient Race"].value_counts().to_dict(),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – MACHINE LEARNING MODEL
# ═════════════════════════════════════════════════════════════════════════════

def build_pipeline() -> Pipeline:
    """Build a scikit-learn Pipeline with preprocessing + RandomForest."""
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUM_FEATURES),
        ("cat", categorical_transformer, CAT_FEATURES),
    ])
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200, max_depth=10,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ))
    ])
    return pipeline


def train_model(df: pd.DataFrame):
    """Train RandomForestRegressor and evaluate it."""
    df_clean = df.dropna(subset=[TARGET_COL] + ALL_FEATURES).copy()
    X = df_clean[ALL_FEATURES]
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    mse    = mean_squared_error(y_test, y_pred)
    rmse   = np.sqrt(mse)
    r2     = r2_score(y_test, y_pred)

    cv_scores     = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_mae  = mean_absolute_error(y_test, baseline_pred)

    ohe_cols = list(
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CAT_FEATURES)
    )
    feature_names = NUM_FEATURES + ohe_cols
    importances   = pipeline.named_steps["regressor"].feature_importances_

    feat_imp = dict(zip(feature_names, [round(float(v), 4) for v in importances]))
    parent_imp = {}
    for fn, iv in feat_imp.items():
        parent = fn.split("_")[0] if "_" in fn else fn
        parent_imp[parent] = round(parent_imp.get(parent, 0) + iv, 4)

    metrics = {
        "model_name"          : "Random Forest Regressor",
        "target_variable"     : TARGET_COL,
        "features_used"       : ALL_FEATURES,
        "train_samples"       : int(len(X_train)),
        "test_samples"        : int(len(X_test)),
        "mae"                 : round(float(mae), 4),
        "mse"                 : round(float(mse), 4),
        "rmse"                : round(float(rmse), 4),
        "r2_score"            : round(float(r2), 4),
        "cv_r2_mean"          : round(float(cv_scores.mean()), 4),
        "cv_r2_std"           : round(float(cv_scores.std()), 4),
        "baseline_mae"        : round(float(baseline_mae), 4),
        "feature_importances" : parent_imp,
    }

    joblib.dump(pipeline, MODEL_PATH)
    print(f"[ML] Model saved -> {MODEL_PATH}")
    print(f"[ML] MAE={mae:.2f} min | RMSE={rmse:.2f} min | R2={r2:.4f}")
    return pipeline, metrics, X_test, y_test, y_pred


def load_or_train(df: pd.DataFrame):
    """Load persisted model if available, otherwise train fresh."""
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        print(f"[ML] Loaded model from {MODEL_PATH}")
        df_clean = df.dropna(subset=[TARGET_COL] + ALL_FEATURES).copy()
        X = df_clean[ALL_FEATURES]
        y = df_clean[TARGET_COL]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        y_pred = pipeline.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        ohe_cols = list(
            pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(CAT_FEATURES)
        )
        feature_names = NUM_FEATURES + ohe_cols
        importances   = pipeline.named_steps["regressor"].feature_importances_
        feat_imp = dict(zip(feature_names, [round(float(v), 4) for v in importances]))
        parent_imp = {}
        for fn, iv in feat_imp.items():
            parent = fn.split("_")[0] if "_" in fn else fn
            parent_imp[parent] = round(parent_imp.get(parent, 0) + iv, 4)

        metrics = {
            "model_name"          : "Random Forest Regressor",
            "target_variable"     : TARGET_COL,
            "features_used"       : ALL_FEATURES,
            "train_samples"       : int(len(X) * 0.8),
            "test_samples"        : int(len(X_test)),
            "mae"                 : round(float(mae), 4),
            "mse"                 : round(float(mean_squared_error(y_test, y_pred)), 4),
            "rmse"                : round(float(rmse), 4),
            "r2_score"            : round(float(r2), 4),
            "cv_r2_mean"          : 0.0,
            "cv_r2_std"           : 0.0,
            "baseline_mae"        : round(float(mean_absolute_error(y_test, np.full(len(y_test), y_test.mean()))), 4),
            "feature_importances" : parent_imp,
        }
        return pipeline, metrics, X_test, y_test, y_pred
    return train_model(df)


def predict_single(pipeline, input_dict: dict) -> float:
    """Predict wait time for a single patient."""
    row = pd.DataFrame([{
        "Patient Age"           : float(input_dict.get("Patient Age", 30)),
        "Patient Gender"        : str(input_dict.get("Patient Gender", "Male")),
        "Patient Race"          : str(input_dict.get("Patient Race", "White")),
        "Department Referral"   : str(input_dict.get("Department Referral", "No Referral")),
        "Patient Admission Flag": str(input_dict.get("Patient Admission Flag", "Not Admission")),
    }])
    pred = pipeline.predict(row[ALL_FEATURES])[0]
    return round(float(pred), 1)


def get_urgency_thresholds(df: pd.DataFrame):
    """
    Compute urgency thresholds from the actual wait-time distribution.
    Uses the 33rd and 66th percentiles so urgency spread varies with
    whatever data is loaded — avoids hardcoded 20/40 that caused
    all predictions to land in 'Medium' for this near-uniform dataset.
    Returns (low_max, medium_max) where:
        <= low_max   -> Low urgency
        <= medium_max-> Medium urgency
        > medium_max -> High urgency
    """
    wait_col = df["Patient Waittime"].dropna()
    p33 = float(np.percentile(wait_col, 33))
    p66 = float(np.percentile(wait_col, 66))
    return round(p33, 1), round(p66, 1)


def get_urgency(pred_wait: float, low_max: float, medium_max: float) -> str:
    """Return 'low', 'medium', or 'high' urgency label."""
    if pred_wait <= low_max:
        return "low"
    if pred_wait <= medium_max:
        return "medium"
    return "high"


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – VISUALISATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

# Shared chart style — crisp, modern, purple-accented
_FONT     = {"family": "DejaVu Sans"}
_TXT      = "#2D2B55"      # dark indigo text on charts
_TXT_MUT  = "#64748B"
_SPINE    = "#E2E0FF"      # very light purple spine
_GRID     = "#F0EDFF"      # faint purple grid

matplotlib.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.weight"      : "bold",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.edgecolor"   : _SPINE,
    "axes.labelcolor"  : _TXT_MUT,
    "xtick.color"      : _TXT_MUT,
    "ytick.color"      : _TXT_MUT,
    "text.color"       : _TXT,
    "figure.facecolor" : BG_CHART,
    "axes.facecolor"   : BG_CHART,
})

def _ax(ax, title="", xlabel="", ylabel="", grid_axis="y"):
    """Apply polished shared styling."""
    ax.set_facecolor(BG_CHART)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=_GRID, linewidth=0.9, linestyle="--", zorder=0)
    ax.spines["left"].set_color(_SPINE)
    ax.spines["bottom"].set_color(_SPINE)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color=_TXT, pad=12, loc="left")
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=_TXT_MUT, labelpad=6)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=_TXT_MUT, labelpad=6)


def plot_waittime_distribution(df: pd.DataFrame):
    vals = df["Patient Waittime"].dropna()
    mean = vals.mean()
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=BG_CHART)
    # gradient-coloured histogram via individual bar coloring
    n, bins, patches = ax.hist(vals, bins=30, edgecolor="white",
                                linewidth=0.4, alpha=1.0)
    # colour bars purple -> cyan by position
    cmap = matplotlib.colormaps["cool"]
    for i, patch in enumerate(patches):
        patch.set_facecolor(cmap(i / len(patches)))
        patch.set_alpha(0.88)
    ax.axvline(mean, color=C_RED, linewidth=2, linestyle="--", zorder=5,
               label=f"Mean  {mean:.1f} min")
    ax.axvspan(mean - 5, mean + 5, alpha=0.06, color=C_RED, zorder=0)
    leg = ax.legend(fontsize=9, framealpha=0, labelcolor=_TXT)
    _ax(ax, "Wait Time Distribution", "Minutes", "Patients")
    fig.tight_layout(pad=1.2)
    return fig


def plot_department_avg_wait(df: pd.DataFrame):
    dw = df.groupby("Department Referral")["Patient Waittime"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=BG_CHART)
    # colour: green = below avg, orange = above
    colours = [C_GREEN if v <= dw.mean() else C_ORANGE for v in dw.values]
    bars = ax.barh(dw.index, dw.values, color=colours,
                   edgecolor="white", height=0.55, zorder=3)
    # add value labels
    for bar, v in zip(bars, dw.values):
        ax.text(v + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=8.5,
                color=_TXT, fontweight="600")
    ax.axvline(dw.mean(), color="#94A3B8", linestyle="--", linewidth=1.3,
               label=f"Overall avg  {dw.mean():.1f} min", zorder=4)
    ax.legend(fontsize=8.5, framealpha=0, labelcolor=_TXT)
    _ax(ax, "Avg Wait Time by Department", "Minutes", grid_axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=1.2)
    return fig


def plot_admission_flag_pie(df: pd.DataFrame):
    counts = df["Patient Admission Flag"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.8), facecolor=BG_CHART)
    wedge_props = dict(edgecolor="white", linewidth=2.5, width=0.6)
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index,
        autopct="%1.1f%%",
        colors=[C_BLUE, C_ORANGE],
        startangle=130,
        wedgeprops=wedge_props,
        pctdistance=0.78,
        shadow=False,
    )
    for t in texts:
        t.set_fontsize(9.5); t.set_color(_TXT); t.set_fontweight("600")
    for at in autotexts:
        at.set_fontsize(9); at.set_fontweight("bold"); at.set_color("white")
    # centre label
    ax.text(0, 0, f"{len(df):,}\nPatients", ha="center", va="center",
            fontsize=8.5, color=_TXT_MUT, fontweight="600", linespacing=1.4)
    ax.set_title("Admission Breakdown", fontsize=11, fontweight="bold",
                 color=_TXT, pad=10, loc="left")
    fig.tight_layout(pad=1.2)
    return fig


def plot_age_group_waittime(df: pd.DataFrame):
    """Box plot – wait time by age group. tick_labels for mpl 3.9+."""
    order   = ["Child (0-12)", "Teen (13-17)", "Young Adult (18-35)",
               "Adult (36-60)", "Senior (60+)"]
    order   = [o for o in order if o in df["Age Group"].unique()]
    data    = [df[df["Age Group"] == g]["Patient Waittime"].dropna().values
               for g in order]
    colours = [C_BLUE, C_PURPLE, C_TEAL, C_ORANGE, C_RED]
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=BG_CHART)
    import matplotlib as _mpl
    mpl_ver = tuple(int(x) for x in _mpl.__version__.split(".")[:2])
    bp_kw = dict(
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(color="#A78BFA", linewidth=1.2),
        capprops=dict(color="#A78BFA", linewidth=1.5),
        flierprops=dict(marker="o", markersize=3.5,
                        markerfacecolor="#C4B5FD", alpha=0.5, linestyle="none"),
        boxprops=dict(linewidth=0),
    )
    bp = (ax.boxplot(data, tick_labels=order, **bp_kw) if mpl_ver >= (3, 9)
          else ax.boxplot(data, labels=order, **bp_kw))
    for patch, c in zip(bp["boxes"], colours[:len(order)]):
        patch.set_facecolor(c); patch.set_alpha(0.82)
    _ax(ax, "Wait Time by Age Group", "Age Group", "Wait Time (min)")
    plt.setp(ax.get_xticklabels(), rotation=10, ha="right", fontsize=8.5)
    fig.tight_layout(pad=1.2)
    return fig


def plot_gender_satisfaction(df: pd.DataFrame):
    gs = df.groupby("Patient Gender")["Patient Satisfaction Score"].mean()
    fig, ax = plt.subplots(figsize=(5, 3.8), facecolor=BG_CHART)
    bar_colors = [C_BLUE, C_ORANGE]
    bars = ax.bar(gs.index, gs.values, color=bar_colors,
                  edgecolor="white", width=0.42, zorder=3)
    ax.set_ylim(0, 11)
    for bar, v in zip(bars, gs.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                f"{v:.2f}", ha="center", fontsize=10.5,
                fontweight="bold", color=_TXT)
    # subtle background stripe
    ax.axhspan(0, gs.mean(), alpha=0.04, color=C_BLUE)
    _ax(ax, "Avg Patient Satisfaction by Gender", "", "Score (0–10)")
    fig.tight_layout(pad=1.2)
    return fig


def plot_hourly_patient_volume(df: pd.DataFrame):
    hourly = df["Admission Hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=BG_CHART)
    # gradient area fill using polygon
    ax.fill_between(hourly.index, hourly.values,
                    alpha=0.18, color=C_BLUE, zorder=1)
    ax.fill_between(hourly.index, hourly.values,
                    alpha=0.06, color=C_PURPLE, zorder=1)
    ax.plot(hourly.index, hourly.values,
            color=C_BLUE, linewidth=2.2, zorder=5)
    ax.scatter(hourly.index, hourly.values,
               color="white", edgecolors=C_BLUE, linewidths=1.8,
               s=28, zorder=6)
    # highlight peak hour
    peak_h = hourly.idxmax()
    ax.scatter([peak_h], [hourly[peak_h]],
               color=C_RED, s=60, zorder=7, label=f"Peak: {peak_h}:00 h")
    ax.legend(fontsize=8.5, framealpha=0, labelcolor=_TXT)
    _ax(ax, "Patient Arrivals by Hour of Day",
        "Hour (0 = midnight)", "Patients", grid_axis="both")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout(pad=1.2)
    return fig


def plot_race_waittime(df: pd.DataFrame):
    rw = df.groupby("Patient Race")["Patient Waittime"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=BG_CHART)
    palette = [C_BLUE, C_PURPLE, C_TEAL, C_ORANGE, C_GREEN, C_RED, "#A78BFA"]
    bars = ax.barh(rw.index, rw.values,
                   color=palette[:len(rw)], edgecolor="white", height=0.55, zorder=3)
    for bar, v in zip(bars, rw.values):
        ax.text(v + 0.25, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=8.5,
                color=_TXT, fontweight="600")
    ax.axvline(rw.mean(), color="#94A3B8", linestyle="--", linewidth=1.2,
               label=f"Avg  {rw.mean():.1f} min", zorder=4)
    ax.legend(fontsize=8.5, framealpha=0, labelcolor=_TXT)
    _ax(ax, "Avg Wait Time by Race / Ethnicity", "Minutes", grid_axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=1.2)
    return fig


def plot_feature_importance(metrics: dict):
    fi = metrics.get("feature_importances", {})
    if not fi:
        return None
    fi_s = dict(sorted(fi.items(), key=lambda x: x[1]))
    max_v = max(fi_s.values())
    colours = [C_ORANGE if v == max_v else C_BLUE for v in fi_s.values()]
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=BG_CHART)
    bars = ax.barh(list(fi_s.keys()), list(fi_s.values()),
                   color=colours, edgecolor="white", height=0.45, zorder=3)
    for bar, v in zip(bars, fi_s.values()):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9,
                color=_TXT, fontweight="600")
    _ax(ax, "Feature Importances — Random Forest", "Importance Score", grid_axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=1.2)
    return fig


def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=BG_CHART)
    # density-coloured scatter
    ax.scatter(y_test, y_pred,
               alpha=0.28, color=C_BLUE,
               edgecolors="none", s=18, zorder=3)
    mn = min(float(y_test.min()), float(y_pred.min()))
    mx = max(float(y_test.max()), float(y_pred.max()))
    ax.plot([mn, mx], [mn, mx], color=C_RED, linewidth=1.8,
            linestyle="--", label="Perfect fit", zorder=5)
    # ±10 min band
    ax.fill_between([mn, mx], [mn - 10, mx - 10], [mn + 10, mx + 10],
                    alpha=0.06, color=C_BLUE, label="±10 min band")
    ax.legend(fontsize=9, framealpha=0, labelcolor=_TXT)
    _ax(ax, "Actual vs Predicted Wait Times",
        "Actual (min)", "Predicted (min)", grid_axis="both")
    fig.tight_layout(pad=1.2)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – FLASK BACKEND API
# ═════════════════════════════════════════════════════════════════════════════

def create_flask_app(pipeline, metrics: dict, summary: dict, df_ref: pd.DataFrame):
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    # Compute percentile-based urgency thresholds once at startup
    _low_max, _med_max = get_urgency_thresholds(df_ref)

    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "status"    : "running",
            "project"   : "Hospital Patient Flow & Operational Efficiency Prediction",
            "author"    : "Sourajit Mondal",
            "program"   : "IBM Bharat / AICTE Data Analytics with AI Internship",
            "endpoints" : ["/", "/model-info", "/dataset", "/predict"],
            "urgency_thresholds": {
                "low_max_min"   : _low_max,
                "medium_max_min": _med_max,
            },
        })

    @app.route("/model-info", methods=["GET"])
    def model_info():
        return jsonify(metrics)

    @app.route("/dataset", methods=["GET"])
    def dataset_info():
        return jsonify(summary)

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            body = request.get_json(force=True)
            if not body:
                return jsonify({"error": "No JSON body received"}), 400
            required = ["Patient Age", "Patient Gender", "Patient Race",
                        "Department Referral", "Patient Admission Flag"]
            missing = [k for k in required if k not in body]
            if missing:
                return jsonify({"error": f"Missing fields: {missing}"}), 400
            age = float(body["Patient Age"])
            if not (0 <= age <= 120):
                return jsonify({"error": "Patient Age must be 0-120"}), 400
            predicted_wait  = predict_single(pipeline, body)
            urgency_key     = get_urgency(predicted_wait, _low_max, _med_max)
            urgency_lbl     = urgency_key.capitalize()
            return jsonify({
                "predicted_wait_time_minutes" : predicted_wait,
                "urgency_level"               : urgency_lbl,
                "urgency_thresholds"          : {"low": _low_max, "medium": _med_max},
                "interpretation": (
                    f"Predicted wait time is {predicted_wait} minutes "
                    f"({urgency_lbl} urgency). "
                    f"Thresholds: Low <= {_low_max} min, "
                    f"Medium <= {_med_max} min, High > {_med_max} min."
                ),
                "input_received": body,
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


def _start_flask_thread(pipeline, metrics, summary, df_ref):
    """Start Flask in a background daemon thread (used by Streamlit)."""
    import requests as req
    try:
        req.get(f"http://127.0.0.1:{FLASK_PORT}/", timeout=1)
        return   # already running
    except Exception:
        pass

    app = create_flask_app(pipeline, metrics, summary, df_ref)
    t = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=FLASK_PORT,
                               debug=False, use_reloader=False),
        daemon=True
    )
    t.start()
    time.sleep(2)   # give Flask a moment to bind


def run_flask(pipeline, metrics: dict, summary: dict, df_ref: pd.DataFrame):
    app = create_flask_app(pipeline, metrics, summary, df_ref)
    print(f"\n[Flask] Starting backend on http://127.0.0.1:{FLASK_PORT}")
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – STREAMLIT FRONTEND
# ═════════════════════════════════════════════════════════════════════════════

def run_streamlit_ui():
    import streamlit as st

    # ── Page config ──────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Hospital Patient Flow | Sourajit Mondal",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Design tokens (Python side — for chart colours only) ─────────────────
    PRIMARY   = "#7C3AED"
    PRIMARY_L = "#8B5CF6"
    CYAN      = "#06B6D4"
    SUCCESS   = "#10B981"
    WARNING   = "#F59E0B"
    DANGER    = "#EF4444"
    NAVY      = "#0F172A"
    NAV_MID   = "#1E293B"

    # ── Global CSS — theme-adaptive via CSS custom properties ─────────────────
    # All colours are defined as CSS variables with light + dark overrides so
    # the UI is readable in light mode, dark mode, and system/auto mode.
    st.markdown("""
    <style>
    /* ════════════════════════════════════════════════════
       1. CSS CUSTOM PROPERTIES  (light defaults)
    ════════════════════════════════════════════════════ */
    :root {
        --c-primary:   #7C3AED;
        --c-primary-l: #8B5CF6;
        --c-cyan:      #06B6D4;
        --c-success:   #10B981;
        --c-warning:   #F59E0B;
        --c-danger:    #EF4444;

        --bg-page:     #F5F7FA;
        --bg-card:     rgba(255,255,255,0.78);
        --bg-card-s:   rgba(255,255,255,0.55);
        --bg-factor:   rgba(255,255,255,0.65);

        --txt-head:    #1E293B;
        --txt-body:    #334155;
        --txt-muted:   #64748B;
        --txt-lite:    #94A3B8;

        --border-c:    rgba(124,58,237,0.18);
        --border-s:    rgba(124,58,237,0.09);
        --divider:     rgba(124,58,237,0.10);

        --shadow-kpi:  0 4px 28px rgba(124,58,237,0.10), 0 1px 4px rgba(0,0,0,0.06);
        --shadow-card: 0 2px 14px rgba(0,0,0,0.06);
    }

    /* ════════════════════════════════════════════════════
       2. DARK MODE OVERRIDES
    ════════════════════════════════════════════════════ */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-page:    #0F0E17;
            --bg-card:    rgba(30,27,50,0.80);
            --bg-card-s:  rgba(40,36,66,0.60);
            --bg-factor:  rgba(30,27,50,0.65);

            --txt-head:   #F1F5F9;
            --txt-body:   #CBD5E1;
            --txt-muted:  #94A3B8;
            --txt-lite:   #64748B;

            --border-c:   rgba(139,92,246,0.28);
            --border-s:   rgba(139,92,246,0.14);
            --divider:    rgba(139,92,246,0.15);

            --shadow-kpi:  0 4px 28px rgba(124,58,237,0.25), 0 1px 4px rgba(0,0,0,0.3);
            --shadow-card: 0 2px 14px rgba(0,0,0,0.3);
        }
    }

    /* ─── Streamlit dark theme class override (when user picks dark in settings) */
    [data-theme="dark"] {
        --bg-page:    #0F0E17;
        --bg-card:    rgba(30,27,50,0.80);
        --bg-card-s:  rgba(40,36,66,0.60);
        --bg-factor:  rgba(30,27,50,0.65);
        --txt-head:   #F1F5F9;
        --txt-body:   #CBD5E1;
        --txt-muted:  #94A3B8;
        --txt-lite:   #64748B;
        --border-c:   rgba(139,92,246,0.28);
        --border-s:   rgba(139,92,246,0.14);
        --divider:    rgba(139,92,246,0.15);
        --shadow-kpi:  0 4px 28px rgba(124,58,237,0.25), 0 1px 4px rgba(0,0,0,0.3);
        --shadow-card: 0 2px 14px rgba(0,0,0,0.3);
    }

    /* ════════════════════════════════════════════════════
       3. BASE LAYOUT
    ════════════════════════════════════════════════════ */
    [data-testid="stAppViewContainer"] {
        background: var(--bg-page) !important;
    }
    [data-testid="stHeader"]  { background: transparent !important; }
    .block-container {
        padding: 1.8rem 2.2rem 2rem 2.2rem;
        max-width: 1400px;
    }
    /* fix Streamlit's own text colour in dark mode */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    [data-testid="stMarkdownContainer"] p {
        color: var(--txt-body) !important;
    }

    /* ════════════════════════════════════════════════════
       4. SIDEBAR  (always dark navy — fixed, not adaptive)
    ════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(170deg,#0F172A 0%,#1E1B3A 60%,#1E293B 100%) !important;
        border-right: 1px solid rgba(124,58,237,0.22) !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] small {
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.09) !important; }

    /* nav radio items */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255,255,255,0.035) !important;
        border-radius: 9px !important;
        padding: 9px 13px !important;
        margin-bottom: 3px !important;
        font-size: 0.9rem !important;
        color: #CBD5E1 !important;
        display: block !important;
        border: 1px solid transparent !important;
        transition: background 0.15s, border-color 0.15s;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(124,58,237,0.18) !important;
        border-color: rgba(124,58,237,0.3) !important;
        color: #F1F5F9 !important;
    }
    /* selected radio item */
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-selected="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] input:checked + label {
        background: rgba(124,58,237,0.28) !important;
        border-color: rgba(124,58,237,0.5) !important;
        color: #F1F5F9 !important;
    }

    /* ════════════════════════════════════════════════════
       5. HERO TITLE  (gradient text — works on both themes)
    ════════════════════════════════════════════════════ */
    .hero-title {
        font-size: 2rem; font-weight: 900;
        background: linear-gradient(135deg,#7C3AED 0%,#06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2; margin-bottom: 4px;
    }
    .hero-sub {
        font-size: 0.93rem;
        color: var(--txt-muted);
        margin-top: 0; margin-bottom: 0;
    }

    /* ════════════════════════════════════════════════════
       6. SECTION HEADER
    ════════════════════════════════════════════════════ */
    .sec-hdr {
        font-size: 1.02rem; font-weight: 700;
        color: var(--txt-head);
        border-left: 4px solid #7C3AED;
        padding-left: 10px;
        margin: 1.6rem 0 0.8rem 0;
        letter-spacing: 0.2px;
    }

    /* ════════════════════════════════════════════════════
       7. KPI GLASS CARDS  (true glassmorphism blobs)
    ════════════════════════════════════════════════════ */
    .kpi-glass {
        background: var(--bg-card);
        backdrop-filter: blur(18px) saturate(160%);
        -webkit-backdrop-filter: blur(18px) saturate(160%);
        border: 1px solid var(--border-c);
        border-radius: 20px;
        padding: 1.3rem 1rem 1.1rem 1rem;
        text-align: center;
        box-shadow: var(--shadow-kpi);
        position: relative;
        overflow: hidden;
        /* subtle inner glow */
        outline: 1px solid rgba(255,255,255,0.12);
        outline-offset: -1px;
    }
    /* top gradient bar */
    .kpi-glass::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg,#7C3AED,#06B6D4);
        border-radius: 20px 20px 0 0;
    }
    /* ambient blob glow behind card */
    .kpi-glass::after {
        content: "";
        position: absolute;
        bottom: -30px; right: -20px;
        width: 80px; height: 80px;
        background: radial-gradient(circle,rgba(124,58,237,0.15),transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .kpi-icon { font-size: 1.55rem; margin-bottom: 5px; display: block; }
    .kpi-val  {
        font-size: 1.75rem; font-weight: 800;
        color: #7C3AED;
        line-height: 1.1; letter-spacing: -0.5px;
    }
    .kpi-lbl  {
        font-size: 0.68rem; color: var(--txt-lite);
        text-transform: uppercase; letter-spacing: 0.9px;
        margin-top: 5px;
    }

    /* ════════════════════════════════════════════════════
       8. CHART GLASS CONTAINER
    ════════════════════════════════════════════════════ */
    .chart-card {
        background: var(--bg-card);
        backdrop-filter: blur(10px) saturate(140%);
        -webkit-backdrop-filter: blur(10px) saturate(140%);
        border: 1px solid var(--border-c);
        border-radius: 16px;
        padding: 1.1rem 1rem 0.6rem 1rem;
        box-shadow: var(--shadow-card);
        margin-bottom: 0.6rem;
        position: relative;
        overflow: hidden;
    }
    /* thin purple left accent */
    .chart-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: linear-gradient(180deg,#7C3AED,#06B6D4);
        border-radius: 16px 0 0 16px;
    }

    /* ════════════════════════════════════════════════════
       9. PREDICTION HERO CARD
    ════════════════════════════════════════════════════ */
    .pred-hero {
        background: var(--bg-card-s);
        backdrop-filter: blur(20px) saturate(150%);
        -webkit-backdrop-filter: blur(20px) saturate(150%);
        border: 1px solid rgba(124,58,237,0.22);
        border-radius: 20px;
        padding: 2.2rem 2rem 1.8rem 2rem;
        text-align: center;
        box-shadow: 0 8px 40px rgba(124,58,237,0.14);
        margin: 1rem 0 1.5rem 0;
        position: relative; overflow: hidden;
    }
    .pred-hero::before {
        content:"";
        position:absolute; top:-60px; right:-60px;
        width:200px; height:200px;
        background:radial-gradient(circle,rgba(124,58,237,0.12),transparent 65%);
        border-radius:50%; pointer-events:none;
    }
    .pred-lbl-sm {
        font-size: 0.72rem; color: var(--txt-lite);
        text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px;
    }
    .pred-num {
        font-size: 3.8rem; font-weight: 900;
        background: linear-gradient(135deg,#7C3AED 0%,#06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1; letter-spacing: -3px;
    }
    .pred-unit {
        font-size: 1.3rem; color: var(--txt-muted);
        font-weight: 500; margin-left: 5px;
    }
    .pred-interp {
        font-size: 0.9rem; color: var(--txt-muted);
        margin-top: 14px; line-height: 1.6;
        max-width: 520px; margin-left: auto; margin-right: auto;
    }

    /* ════════════════════════════════════════════════════
       10. URGENCY BADGES
    ════════════════════════════════════════════════════ */
    .badge {
        display: inline-block;
        padding: 5px 18px; border-radius: 24px;
        font-size: 0.78rem; font-weight: 700;
        letter-spacing: 0.5px; margin-top: 12px;
    }
    .badge-low    { background:rgba(16,185,129,0.18); color:#10B981;
                   border:1px solid rgba(16,185,129,0.35); }
    .badge-medium { background:rgba(245,158,11,0.18); color:#F59E0B;
                   border:1px solid rgba(245,158,11,0.35); }
    .badge-high   { background:rgba(239,68,68,0.18);  color:#EF4444;
                   border:1px solid rgba(239,68,68,0.35); }

    /* ════════════════════════════════════════════════════
       11. FACTOR TABLE
    ════════════════════════════════════════════════════ */
    .factor-table {
        background: var(--bg-factor);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-s);
        border-radius: 14px; padding: 1rem 1.3rem;
        margin-top: 1.3rem;
    }
    .factor-row {
        display:flex; justify-content:space-between;
        padding: 7px 0;
        border-bottom: 1px solid var(--divider);
        font-size: 0.9rem;
    }
    .factor-row:last-child { border-bottom: none; }
    .factor-key   { color: var(--txt-muted); }
    .factor-value { color: var(--txt-head); font-weight: 600; }

    /* ════════════════════════════════════════════════════
       12. WARNING ALERT
    ════════════════════════════════════════════════════ */
    .alert-warn {
        background: rgba(245,158,11,0.09);
        border: 1px solid rgba(245,158,11,0.28);
        border-left: 4px solid #F59E0B;
        border-radius: 10px;
        padding: 0.95rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: var(--txt-body);
        line-height: 1.6;
    }
    .alert-warn strong { color: var(--txt-head); }

    /* ════════════════════════════════════════════════════
       13. MODEL METRIC CARD
    ════════════════════════════════════════════════════ */
    .metric-card {
        background: var(--bg-card);
        backdrop-filter: blur(14px) saturate(150%);
        -webkit-backdrop-filter: blur(14px) saturate(150%);
        border: 1px solid var(--border-c);
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: var(--shadow-kpi);
        position: relative; overflow: hidden;
    }
    .metric-card::before {
        content:"";
        position:absolute; top:0; left:0; right:0; height:3px;
        background: linear-gradient(90deg,#7C3AED,#06B6D4);
        border-radius: 16px 16px 0 0;
    }
    .metric-val {
        font-size: 1.7rem; font-weight: 800;
    }
    .metric-lbl {
        font-size: 0.68rem; color: var(--txt-lite);
        text-transform: uppercase; letter-spacing: 0.9px;
        margin-top: 5px;
    }

    /* ════════════════════════════════════════════════════
       14. MISC OVERRIDES
    ════════════════════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important; overflow: hidden;
        box-shadow: var(--shadow-card);
    }
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg,#7C3AED 0%,#8B5CF6 100%) !important;
        border: none !important; border-radius: 11px !important;
        font-weight: 700 !important; letter-spacing: 0.4px !important;
        box-shadow: 0 4px 18px rgba(124,58,237,0.35) !important;
        color: #fff !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
    }
    [data-testid="baseButton-primary"]:hover {
        transform: scale(1.04) !important;
        box-shadow: 0 6px 24px rgba(124,58,237,0.5) !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--border-c) !important;
        border-radius: 12px !important;
        background: var(--bg-card-s) !important;
    }

    /* ════════════════════════════════════════════════════
       15. HOVER ZOOM EFFECTS
       All interactive cards scale up 2-4% on hover.
       transition keeps it smooth (200ms ease-out).
    ════════════════════════════════════════════════════ */
    .kpi-glass {
        transition: transform 0.2s ease-out, box-shadow 0.2s ease-out;
        cursor: default;
    }
    .kpi-glass:hover {
        transform: scale(1.045) translateY(-3px);
        box-shadow: 0 10px 36px rgba(124,58,237,0.18),
                    0 2px 8px rgba(0,0,0,0.10) !important;
    }

    .chart-card {
        transition: transform 0.2s ease-out, box-shadow 0.2s ease-out;
    }
    .chart-card:hover {
        transform: scale(1.02) translateY(-2px);
        box-shadow: 0 8px 30px rgba(124,58,237,0.13),
                    0 2px 8px rgba(0,0,0,0.08) !important;
    }

    .metric-card {
        transition: transform 0.2s ease-out, box-shadow 0.2s ease-out;
        cursor: default;
    }
    .metric-card:hover {
        transform: scale(1.045) translateY(-3px);
        box-shadow: 0 10px 36px rgba(124,58,237,0.18),
                    0 2px 8px rgba(0,0,0,0.10) !important;
    }

    .factor-table {
        transition: transform 0.18s ease-out, box-shadow 0.18s ease-out;
    }
    .factor-table:hover {
        transform: scale(1.015);
        box-shadow: 0 6px 22px rgba(124,58,237,0.10) !important;
    }

    .pred-hero {
        transition: transform 0.2s ease-out, box-shadow 0.2s ease-out;
    }
    .pred-hero:hover {
        transform: scale(1.015) translateY(-2px);
        box-shadow: 0 14px 48px rgba(124,58,237,0.18) !important;
    }

    /* ════════════════════════════════════════════════════
       16. DARK / LIGHT THEME ROBUSTNESS
       Streamlit injects .st-emotion-cache-* classes and
       also sets data-theme on <html> or <body>.
       We target ALL known patterns to ensure text stays
       readable in every theme mode.
    ════════════════════════════════════════════════════ */

    /* --- Streamlit dark theme: targets stApp wrapper --- */
    .stApp[data-theme="dark"] {
        --bg-page:    #0F0E17;
        --bg-card:    rgba(30,27,50,0.82);
        --bg-card-s:  rgba(40,36,66,0.62);
        --bg-factor:  rgba(30,27,50,0.68);
        --txt-head:   #F1F5F9;
        --txt-body:   #CBD5E1;
        --txt-muted:  #94A3B8;
        --txt-lite:   #64748B;
        --border-c:   rgba(139,92,246,0.30);
        --border-s:   rgba(139,92,246,0.15);
        --divider:    rgba(139,92,246,0.16);
        --shadow-kpi:  0 4px 28px rgba(124,58,237,0.28), 0 1px 4px rgba(0,0,0,0.35);
        --shadow-card: 0 2px 14px rgba(0,0,0,0.35);
    }

    /* Streamlit also sets it on html element */
    html[data-theme="dark"],
    body[data-theme="dark"] {
        --bg-page:    #0F0E17;
        --bg-card:    rgba(30,27,50,0.82);
        --bg-card-s:  rgba(40,36,66,0.62);
        --bg-factor:  rgba(30,27,50,0.68);
        --txt-head:   #F1F5F9;
        --txt-body:   #CBD5E1;
        --txt-muted:  #94A3B8;
        --txt-lite:   #64748B;
        --border-c:   rgba(139,92,246,0.30);
        --border-s:   rgba(139,92,246,0.15);
        --divider:    rgba(139,92,246,0.16);
        --shadow-kpi:  0 4px 28px rgba(124,58,237,0.28), 0 1px 4px rgba(0,0,0,0.35);
        --shadow-card: 0 2px 14px rgba(0,0,0,0.35);
    }

    /* Force all our custom text classes to use CSS vars (overrides Streamlit defaults) */
    .hero-sub, .sec-hdr, .pred-interp, .pred-lbl-sm,
    .factor-key, .factor-value, .kpi-lbl, .metric-lbl,
    .alert-warn {
        color: var(--txt-body) !important;
    }
    .hero-title { color: transparent !important; } /* keep gradient */
    .sec-hdr    { color: var(--txt-head) !important; }
    .kpi-val    { color: #7C3AED !important; }   /* purple always readable */
    .metric-val { color: inherit; }               /* set per-card inline */
    .factor-value { color: var(--txt-head) !important; font-weight: 600; }

    /* Markdown paragraphs inside our components */
    .stMarkdown p, [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: var(--txt-body) !important;
    }

    /* JSON viewer text visibility in dark mode */
    [data-testid="stJson"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-c) !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Load default data & model (cached) ───────────────────────────────────
    @st.cache_data(show_spinner="Loading dataset...")
    def get_data():
        raw       = load_raw_data(CSV_PATH)
        processed = preprocess_data(raw)
        summary   = get_dataset_summary(processed)
        return processed, summary

    @st.cache_resource(show_spinner="Training model...")
    def get_model():
        raw = load_raw_data(CSV_PATH)
        df  = preprocess_data(raw)
        return load_or_train(df)

    df_default, summary_default = get_data()
    pipeline, metrics, X_test, y_test, y_pred = get_model()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"<div style='padding:16px 4px 8px 4px;'>"
            f"<div style='font-size:1.8rem;'>🏥</div>"
            f"<div style='font-size:1.1rem;font-weight:800;color:#F1F5F9;"
            f"margin:4px 0 2px 0;letter-spacing:0.3px;'>"
            f"Hospital Patient Flow</div>"
            f"<div style='font-size:0.78rem;color:#94A3B8;letter-spacing:0.5px;'>"
            f"Operational Analytics</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        page = st.radio(
            "nav",
            ["📊  Dashboard",
             "📈  Patient Analytics",
             "🔮  Wait-Time Prediction",
             "🤖  Model Performance"],
            label_visibility="collapsed",
        )
        st.divider()

        # ── CSV UPLOAD SECTION ────────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.72rem;color:#94A3B8;text-transform:uppercase;"
            "letter-spacing:0.8px;margin-bottom:6px;'>Upload Your CSV Data</div>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Drop a CSV file here",
            type=["csv"],
            help=(
                "Upload any hospital patient CSV. The app will auto-detect columns "
                "matching: Patient Waittime, Patient Gender, Patient Age, "
                "Department Referral, Patient Admission Flag, Patient Race, "
                "Patient Satisfaction Score."
            ),
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                _uploaded_raw = pd.read_csv(uploaded_file)
                # ── Column auto-mapping ───────────────────────────────────────
                # Maps common naming variants to our standard column names so
                # companies with different CSV schemas can still use the app.
                _col_map = {}
                _norm = {c.lower().replace(" ", "").replace("_", ""): c
                         for c in _uploaded_raw.columns}
                _targets = {
                    "waittime"          : "Patient Waittime",
                    "patientwaittime"   : "Patient Waittime",
                    "wait"              : "Patient Waittime",
                    "waittimemin"       : "Patient Waittime",
                    "gender"            : "Patient Gender",
                    "patientgender"     : "Patient Gender",
                    "sex"               : "Patient Gender",
                    "age"               : "Patient Age",
                    "patientage"        : "Patient Age",
                    "department"        : "Department Referral",
                    "dept"              : "Department Referral",
                    "departmentreferral": "Department Referral",
                    "admissionflag"     : "Patient Admission Flag",
                    "patientadmissionflag": "Patient Admission Flag",
                    "admitted"          : "Patient Admission Flag",
                    "admissionstatus"   : "Patient Admission Flag",
                    "race"              : "Patient Race",
                    "patientrace"       : "Patient Race",
                    "ethnicity"         : "Patient Race",
                    "satisfaction"      : "Patient Satisfaction Score",
                    "satisfactionscore" : "Patient Satisfaction Score",
                    "patientsatisfactionscore": "Patient Satisfaction Score",
                }
                for norm_key, std_col in _targets.items():
                    if norm_key in _norm and std_col not in _uploaded_raw.columns:
                        _col_map[_norm[norm_key]] = std_col

                _uploaded_mapped = _uploaded_raw.rename(columns=_col_map)

                # Ensure required columns exist (fill with sensible defaults if absent)
                for _req_col, _default in [
                    ("Patient Waittime",          30),
                    ("Patient Age",               35),
                    ("Patient Gender",            "Unknown"),
                    ("Patient Race",              "Unknown"),
                    ("Department Referral",       "No Referral"),
                    ("Patient Admission Flag",    "Not Admission"),
                    ("Patient Satisfaction Score", 5),
                    ("Patient Id",                "N/A"),
                ]:
                    if _req_col not in _uploaded_mapped.columns:
                        _uploaded_mapped[_req_col] = _default

                # Add admission date/time stubs if absent
                if "Patient Admission Date" not in _uploaded_mapped.columns:
                    _uploaded_mapped["Patient Admission Date"] = pd.Timestamp.today()
                if "Patient Admission Time" not in _uploaded_mapped.columns:
                    _uploaded_mapped["Patient Admission Time"] = "12:00:00 PM"

                _df_upload   = preprocess_data(_uploaded_mapped)
                _sum_upload  = get_dataset_summary(_df_upload)
                st.session_state["custom_df"]      = _df_upload
                st.session_state["custom_summary"] = _sum_upload
                st.session_state["custom_name"]    = uploaded_file.name
                st.success(f"Loaded: {uploaded_file.name} ({len(_df_upload):,} rows)")
            except Exception as _e:
                st.error(f"Could not parse CSV: {_e}")
                st.session_state.pop("custom_df", None)
        else:
            # Clear custom data when no file is uploaded
            st.session_state.pop("custom_df", None)
            st.session_state.pop("custom_summary", None)
            st.session_state.pop("custom_name", None)

        # ── Resolve active dataset ────────────────────────────────────────────
        # If a custom CSV was uploaded successfully, use it for all pages;
        # otherwise fall back to the default project dataset.
        if "custom_df" in st.session_state:
            df      = st.session_state["custom_df"]
            summary = st.session_state["custom_summary"]
            _data_source = st.session_state["custom_name"]
        else:
            df      = df_default
            summary = summary_default
            _data_source = "Default dataset"

        st.divider()
        st.markdown(
            f"<div style='font-size:0.78rem;line-height:2;padding:0 2px;'>"
            f"<span style='color:#94A3B8;text-transform:uppercase;"
            f"letter-spacing:0.8px;font-size:0.65rem;'>Data Source</span><br>"
            f"<span style='color:#A78BFA;font-weight:700;font-size:0.75rem;"
            f"word-break:break-word;'>{_data_source}</span><br><br>"
            f"<span style='color:#94A3B8;text-transform:uppercase;"
            f"letter-spacing:0.8px;font-size:0.65rem;'>Dataset</span><br>"
            f"<span style='color:#E2E8F0;font-weight:700;'>"
            f"{summary['total_records']:,} patients</span><br><br>"
            f"<span style='color:#94A3B8;text-transform:uppercase;"
            f"letter-spacing:0.8px;font-size:0.65rem;'>Model</span><br>"
            f"<span style='color:#E2E8F0;font-weight:700;'>Random Forest</span><br><br>"
            f"<span style='color:#94A3B8;text-transform:uppercase;"
            f"letter-spacing:0.8px;font-size:0.65rem;'>MAE</span><br>"
            f"<span style='color:#E2E8F0;font-weight:700;'>{metrics['mae']} min</span><br><br>"
            f"<span style='color:#94A3B8;text-transform:uppercase;"
            f"letter-spacing:0.8px;font-size:0.65rem;'>Author</span><br>"
            f"<span style='color:#E2E8F0;font-weight:700;'>Sourajit Mondal</span><br>"
            f"<span style='color:#64748B;font-size:0.72rem;'>"
            f"IBM Bharat / AICTE</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Start Flask backend using active (default) data
    _start_flask_thread(pipeline, metrics, summary_default, df_default)

    # normalise page key (strip emoji prefix)
    pg = page.split("  ", 1)[-1] if "  " in page else page

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE 1 – DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    if pg == "Dashboard":
        # Hero header
        st.markdown(
            f"<div class='hero-title'>Hospital Patient Flow &amp; Operational Efficiency</div>"
            f"<p class='hero-sub'>Monitor patient volume, waiting times, admissions and "
            f"satisfaction using data analytics and machine learning. &nbsp;|&nbsp; "
            f"IBM Bharat / AICTE Internship &nbsp;|&nbsp; Sourajit Mondal</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # KPI glass cards
        k1, k2, k3, k4, k5 = st.columns(5)
        kpi_items = [
            (k1, "👥", f"{summary['total_records']:,}", "Total Patients"),
            (k2, "⏱️", f"{summary['avg_wait_time_min']} min", "Avg Wait Time"),
            (k3, "🏥", f"{summary['admission_rate_pct']}%", "Admission Rate"),
            (k4, "⭐", f"{summary['avg_satisfaction']}/10", "Avg Satisfaction"),
            (k5, "📅", f"{summary['avg_patient_age']} yrs", "Avg Age"),
        ]
        for col, icon, val, lbl in kpi_items:
            with col:
                st.markdown(
                    f"<div class='kpi-glass'>"
                    f"<div class='kpi-icon'>{icon}</div>"
                    f"<div class='kpi-val'>{val}</div>"
                    f"<div class='kpi-lbl'>{lbl}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div class='sec-hdr'>Patient Flow Overview</div>",
                    unsafe_allow_html=True)

        ov1, ov2 = st.columns(2)
        with ov1:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.pyplot(plot_waittime_distribution(df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with ov2:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.pyplot(plot_department_avg_wait(df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-hdr'>Dataset Overview</div>",
                    unsafe_allow_html=True)
        col_a, col_b = st.columns([3, 2])
        with col_a:
            col_info = pd.DataFrame({
                "Column"  : ["Patient Id","Patient Admission Date","Patient Admission Time",
                             "Merged","Patient Gender","Patient Age","Patient Race",
                             "Department Referral","Patient Admission Flag",
                             "Patient Satisfaction Score","Patient Waittime"],
                "Type"    : ["ID","Date","Time","Text","Categorical","Numerical",
                             "Categorical","Categorical","Categorical",
                             "Numerical (sparse)","Numerical — TARGET"],
                "Notes"   : ["Unique ID","Visit date","Arrival time","Doctor name",
                             "Male/Female","Age in years","Race/ethnicity",
                             "Dept or No Referral","Admission or Not",
                             "0–10, ~80% missing","Wait time in minutes"],
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True, height=330)
        with col_b:
            miss_raw = load_raw_data()
            miss_df  = pd.DataFrame({
                "Column"  : ["Satisfaction Score","Patient Waittime","All others"],
                "Missing" : [int(miss_raw["Patient Satisfaction Score"].isnull().sum()),
                             int(miss_raw["Patient Waittime"].isnull().sum()), 0],
                "Fix"     : ["Median fill","Median fill","—"],
            })
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
            dept_df = pd.DataFrame(
                list(summary["top_departments"].items()),
                columns=["Department", "Patients"],
            ).sort_values("Patients", ascending=False)
            st.dataframe(dept_df, use_container_width=True, hide_index=True)

        st.markdown("<div class='sec-hdr'>Data Preview</div>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE 2 – ANALYTICS
    # ══════════════════════════════════════════════════════════════════════════
    elif pg == "Patient Analytics":
        st.markdown(
            f"<div class='hero-title'>Patient Analytics</div>"
            f"<p class='hero-sub'>Explore patterns in wait times, admissions, "
            f"department load, and patient demographics.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("<div class='sec-hdr'>Wait Time Distribution</div>",
                        unsafe_allow_html=True)
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.pyplot(plot_waittime_distribution(df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with r1c2:
            st.markdown("<div class='sec-hdr'>Avg Wait by Department</div>",
                        unsafe_allow_html=True)
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.pyplot(plot_department_avg_wait(df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("<div class='sec-hdr'>Admission Breakdown</div>",
                        unsafe_allow_html=True)
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.pyplot(plot_admission_flag_pie(df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with r2c2:
            st.markdown("<div class='sec-hdr'>Avg Satisfaction by Gender</div>",
                        unsafe_allow_html=True)
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.pyplot(plot_gender_satisfaction(df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-hdr'>Wait Time by Age Group</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.pyplot(plot_age_group_waittime(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            st.markdown("<div class='sec-hdr'>Arrivals by Hour of Day</div>",
                        unsafe_allow_html=True)
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.pyplot(plot_hourly_patient_volume(df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with r3c2:
            st.markdown("<div class='sec-hdr'>Avg Wait by Race / Ethnicity</div>",
                        unsafe_allow_html=True)
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.pyplot(plot_race_waittime(df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-hdr'>Department-level Summary</div>",
                    unsafe_allow_html=True)
        dept_agg = (
            df.groupby("Department Referral")
            .agg(
                Patients           = ("Patient Id", "count"),
                Avg_Wait_Min       = ("Patient Waittime", "mean"),
                Admission_Rate_Pct = ("Patient Admission Flag",
                                      lambda x: round((x == "Admission").mean()*100, 1)),
                Avg_Satisfaction   = ("Patient Satisfaction Score", "mean"),
            )
            .round(2).reset_index().sort_values("Patients", ascending=False)
        )
        st.dataframe(dept_agg, use_container_width=True, hide_index=True)

        st.markdown("<div class='sec-hdr'>Descriptive Statistics</div>",
                    unsafe_allow_html=True)
        st.dataframe(
            df[["Patient Age","Patient Waittime","Patient Satisfaction Score"]]
            .describe().round(2), use_container_width=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE 3 – PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    elif pg == "Wait-Time Prediction":
        st.markdown(
            f"<div class='hero-title'>Wait-Time Prediction</div>"
            f"<p class='hero-sub'>Enter patient details — the Random Forest model "
            f"estimates how long this patient will wait before being seen.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        gender_opts = sorted(df["Patient Gender"].dropna().unique().tolist())
        race_opts   = sorted(df["Patient Race"].dropna().unique().tolist())
        dept_opts   = sorted(df["Department Referral"].dropna().unique().tolist())
        flag_opts   = sorted(df["Patient Admission Flag"].dropna().unique().tolist())

        with st.form("predict_form"):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                age    = st.number_input("Patient Age (years)",
                                          min_value=0, max_value=120,
                                          value=35, step=1)
                gender = st.selectbox("Patient Gender", gender_opts)
            with fc2:
                race   = st.selectbox("Race / Ethnicity", race_opts)
                dept   = st.selectbox("Department Referral", dept_opts)
            with fc3:
                flag   = st.selectbox("Admission Status", flag_opts)
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button(
                    "🔮  Predict Wait Time",
                    type="primary", use_container_width=True,
                )

        # Compute percentile-based urgency thresholds from active dataset
        _urg_low, _urg_med = get_urgency_thresholds(df)

        if submitted:
            inp = {
                "Patient Age"           : age,
                "Patient Gender"        : gender,
                "Patient Race"          : race,
                "Department Referral"   : dept,
                "Patient Admission Flag": flag,
            }
            pred_wait   = predict_single(pipeline, inp)
            urgency_key = get_urgency(pred_wait, _urg_low, _urg_med)
            urgency_txt = {"low":   "Short waiting time — patient should be seen soon.",
                           "medium":"Moderate waiting time — standard queue applies.",
                           "high":  "Longer wait expected — consider priority review."}[urgency_key]
            urgency_lbl = {"low":"Low","medium":"Medium","high":"High"}[urgency_key]

            interp = (
                f"The model predicts an estimated waiting time of approximately "
                f"<strong>{pred_wait} minutes</strong> based on the selected patient "
                f"and hospital characteristics."
            )

            # Big prediction hero card
            st.markdown(
                f"<div class='pred-hero'>"
                f"<div class='pred-lbl-sm'>Predicted Waiting Time</div>"
                f"<div><span class='pred-num'>{pred_wait}</span>"
                f"<span class='pred-unit'>min</span></div>"
                f"<div><span class='badge badge-{urgency_key}'>"
                f"{urgency_lbl} Urgency — {urgency_txt}</span></div>"
                f"<div class='pred-interp'>{interp}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Factor breakdown table
            st.markdown(
                f"<div class='factor-table'>"
                f"<div style='font-size:0.78rem;color:#94A3B8;text-transform:uppercase;"
                f"letter-spacing:0.8px;margin-bottom:8px;'>Input Factors</div>"
                f"<div class='factor-row'><span class='factor-key'>Age</span>"
                f"<span class='factor-value'>{age} years</span></div>"
                f"<div class='factor-row'><span class='factor-key'>Gender</span>"
                f"<span class='factor-value'>{gender}</span></div>"
                f"<div class='factor-row'><span class='factor-key'>Race / Ethnicity</span>"
                f"<span class='factor-value'>{race}</span></div>"
                f"<div class='factor-row'><span class='factor-key'>Department</span>"
                f"<span class='factor-value'>{dept}</span></div>"
                f"<div class='factor-row'><span class='factor-key'>Patient Type</span>"
                f"<span class='factor-value'>{flag}</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Flask API call
            try:
                import requests as req_lib
                resp = req_lib.post(
                    f"http://127.0.0.1:{FLASK_PORT}/predict",
                    json=inp, timeout=3,
                )
                if resp.status_code == 200:
                    with st.expander("Flask API JSON Response", expanded=False):
                        st.json(resp.json())
            except Exception:
                pass

        st.markdown("<div class='sec-hdr'>Example Predictions</div>",
                    unsafe_allow_html=True)
        st.caption(
            f"Urgency thresholds (data-driven): "
            f"Low <= {_urg_low} min | Medium <= {_urg_med} min | High > {_urg_med} min"
        )
        examples = [
            {"Patient Age":25,"Patient Gender":"Female","Patient Race":"White",
             "Department Referral":"No Referral","Patient Admission Flag":"Not Admission"},
            {"Patient Age":70,"Patient Gender":"Male","Patient Race":"African American",
             "Department Referral":"General Practice","Patient Admission Flag":"Admission"},
            {"Patient Age":45,"Patient Gender":"Female","Patient Race":"Asian",
             "Department Referral":"Orthopedics","Patient Admission Flag":"Not Admission"},
            {"Patient Age":8, "Patient Gender":"Male","Patient Race":"White",
             "Department Referral":"Physiotherapy","Patient Admission Flag":"Not Admission"},
        ]
        ex_rows = []
        for e in examples:
            pw    = predict_single(pipeline, e)
            urg_k = get_urgency(pw, _urg_low, _urg_med)
            ex_rows.append({**e, "Predicted Wait (min)": pw,
                            "Urgency": urg_k.capitalize()})
        st.dataframe(pd.DataFrame(ex_rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE 4 – MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════════
    elif pg == "Model Performance":
        st.markdown(
            f"<div class='hero-title'>Model Performance</div>"
            f"<p class='hero-sub'>Evaluation metrics and diagnostics for the "
            f"trained Random Forest Regressor.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Metric glass cards
        mc1, mc2, mc3, mc4 = st.columns(4)
        r2_val = metrics["r2_score"]
        r2_color = SUCCESS if r2_val > 0.3 else (WARNING if r2_val > 0 else DANGER)
        for col, val, lbl, color in [
            (mc1, r2_val,                  "R\u00b2 Score",   r2_color),
            (mc2, f"{metrics['mae']} min", "MAE",             PRIMARY),
            (mc3, f"{metrics['rmse']} min","RMSE",            PRIMARY),
            (mc4, f"{metrics.get('baseline_mae','N/A')} min", "Baseline MAE", "#64748B"),
        ]:
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-val' style='color:{color};'>{val}</div>"
                    f"<div class='metric-lbl'>{lbl}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Honest model warning
        st.markdown(
            f"<div class='alert-warn'>"
            f"<strong>⚠️ Model Limitation Notice</strong><br>"
            f"The current R\u00b2 score of <strong>{r2_val}</strong> indicates limited "
            f"predictive performance. This is expected: the target variable "
            f"(Patient Waittime) is approximately uniformly distributed between 0–60 "
            f"minutes, making variance explanation inherently difficult. "
            f"The model still provides a useful MAE of <strong>{metrics['mae']} min</strong> "
            f"and meaningfully outperforms the naive mean-predictor baseline "
            f"({metrics.get('baseline_mae','N/A')} min). "
            f"Predictions should be interpreted as estimates, not exact values.</div>",
            unsafe_allow_html=True,
        )

        cp1, cp2 = st.columns(2)
        with cp1:
            st.markdown("<div class='sec-hdr'>Actual vs Predicted</div>",
                        unsafe_allow_html=True)
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.pyplot(plot_actual_vs_predicted(y_test, y_pred),
                      use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with cp2:
            fi_fig = plot_feature_importance(metrics)
            if fi_fig:
                st.markdown("<div class='sec-hdr'>Feature Importances</div>",
                            unsafe_allow_html=True)
                st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                st.pyplot(fi_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-hdr'>Full Evaluation Metrics</div>",
                    unsafe_allow_html=True)
        metrics_show = {k: v for k, v in metrics.items()
                        if k not in ("features_used","feature_importances")}
        st.json(metrics_show)

        st.markdown("<div class='sec-hdr'>Model Configuration</div>",
                    unsafe_allow_html=True)
        cfg = pd.DataFrame({
            "Parameter": ["Algorithm","n_estimators","max_depth","min_samples_split",
                          "min_samples_leaf","random_state","Preprocessing",
                          "Test Split","CV Folds"],
            "Value":     ["Random Forest Regressor",200,10,5,2,42,
                          "OneHotEncoder + StandardScaler","20%",5],
        })
        st.dataframe(cfg, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 6 – ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Hospital Patient Flow & Operational Efficiency Prediction")
    print("  Author : Sourajit Mondal")
    print("  IBM Bharat / AICTE Data Analytics with AI Internship")
    print("=" * 70)

    print("\n[Data] Loading dataset ...")
    df_raw  = load_raw_data(CSV_PATH)
    df      = preprocess_data(df_raw)
    summary = get_dataset_summary(df)
    print(f"[Data] {summary['total_records']:,} records | "
          f"{summary['total_columns']} columns | "
          f"Missing: {summary['missing_values']} | "
          f"Duplicates: {summary['duplicate_records']}")

    print("\n[ML] Training model ...")
    pipeline, metrics, X_test, y_test, y_pred = train_model(df)

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "flask"
    if mode == "train":
        print("\n[Done] Training complete. Model saved.")
        return

    run_flask(pipeline, metrics, summary, df)


def _is_streamlit():
    """Return True when executed by `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        pass
    if len(sys.argv) >= 1:
        if "streamlit" in sys.argv[0].replace("\\", "/").lower():
            return True
    return False


if __name__ == "__main__":
    if _is_streamlit():
        run_streamlit_ui()
    else:
        main()
