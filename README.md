# Hospital Patient Flow and Operational Efficiency Prediction Using Machine Learning

**Author:** Sourajit Mondal  
**Program:** IBM Bharat / AICTE Data Analytics with AI Virtual Internship  
**Submission File:** `SourajitMondal_HospitalPatientFlow.py`

---

## Project Description

This project analyses hospital patient flow data to understand operational patterns and predict patient waiting times using machine learning. It covers the complete analytics pipeline — from raw data ingestion and exploratory analysis to model training, a REST API backend, and an interactive web dashboard.

---

## Problem Statement

Hospitals worldwide face the challenge of managing unpredictable patient volumes, long waiting times, and inefficient resource allocation. Accurate prediction of patient waiting times enables hospitals to:

- Allocate staff proactively based on expected demand
- Reduce patient dissatisfaction caused by long waits
- Identify high-load departments needing priority attention
- Support data-driven operational decision-making

---

## Objectives

1. Perform exploratory data analysis on hospital patient flow records
2. Identify key operational metrics (wait time, admission rate, satisfaction)
3. Build a machine learning model to predict patient waiting time
4. Expose the trained model through a REST API (Flask)
5. Build an interactive analytics dashboard (Streamlit)
6. Provide actionable insights for hospital administrators

---

## Dataset Description

| Field | Description |
|-------|-------------|
| Patient Id | Unique patient identifier |
| Patient Admission Date | Date of hospital visit (MM/DD/YYYY) |
| Patient Admission Time | Arrival time (HH:MM:SS AM/PM) |
| Merged | Doctor/staff name associated with the visit |
| Patient Gender | Male / Female |
| Patient Age | Age in years (0–100+) |
| Patient Race | Race/ethnicity of the patient |
| Department Referral | Referred department (or "None" = No Referral) |
| Patient Admission Flag | "Admission" or "Not Admission" |
| Patient Satisfaction Score | 0–10 satisfaction rating (sparse – ~80% missing) |
| Patient Waittime | **TARGET** – wait time in minutes (0–60) |

**Target Variable:** `Patient Waittime`  
**Reason:** Numeric, well-distributed, directly actionable for operations.

### Kaggle Dataset

[Healthcare Analytics – Patient Flow Data on Kaggle](https://www.kaggle.com/datasets)

---

## Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| Data Analytics | Pandas, NumPy, Matplotlib, Seaborn |
| Machine Learning | scikit-learn (Random Forest Regressor) |
| API Backend | Flask |
| Frontend Dashboard | Streamlit |
| Model Persistence | joblib |
| HTTP Client | requests |

---

## Python Libraries

```
pandas        – data loading, cleaning, aggregation
numpy         – numerical operations
scikit-learn  – ML pipeline, preprocessing, metrics
flask         – REST API backend
streamlit     – interactive web dashboard
matplotlib    – visualisations
seaborn       – statistical plots
requests      – Streamlit → Flask API calls
joblib        – model serialisation
```

---

## Project Directory Structure

```
Hospital_Patient_Flow_Project/
│
├── healthcare_analytics_patient_flow_data.csv   ← dataset
├── SourajitMondal_HospitalPatientFlow.py        ← SINGLE source file
├── requirements.txt
├── README.md
├── SourajitMondal_ProjectReport.docx
└── trained_model.joblib                         ← auto-generated after first run
```

> **Important:** There is **only one** Python source file. The data loading,
> preprocessing, ML training, Flask API, and Streamlit UI are all implemented
> inside `SourajitMondal_HospitalPatientFlow.py` as clearly separated functions
> and sections. The entry-point dispatcher (`main()` / `run_streamlit_ui()`)
> selects the correct mode based on how the file is launched.

---

## Installation

### 1. Clone / copy the project folder

```bash
cd Hospital_Patient_Flow_Project
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option A — Run Flask backend (trains model + starts API)

```bash
python SourajitMondal_HospitalPatientFlow.py flask
```

Or simply:

```bash
python SourajitMondal_HospitalPatientFlow.py
```

Flask starts on **http://127.0.0.1:5000**

---

### Option B — Run Streamlit dashboard

```bash
streamlit run SourajitMondal_HospitalPatientFlow.py
```

Streamlit opens in your browser at **http://localhost:8501**

---

### Option C — Train model only (no server)

```bash
python SourajitMondal_HospitalPatientFlow.py train
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Project info & status |
| GET | `/model-info` | Model name, target, evaluation metrics |
| GET | `/dataset` | Dataset summary (rows, columns, KPIs) |
| POST | `/predict` | Predict wait time for a patient |

---

## Example API Requests

### GET /

```bash
curl http://127.0.0.1:5000/
```

**Response:**
```json
{
  "status": "running",
  "project": "Hospital Patient Flow & Operational Efficiency Prediction",
  "author": "Sourajit Mondal",
  "program": "IBM Bharat / AICTE Data Analytics with AI Internship",
  "flask_port": 5000,
  "endpoints": ["/", "/model-info", "/dataset", "/predict"]
}
```

---

### GET /model-info

```bash
curl http://127.0.0.1:5000/model-info
```

**Response:**
```json
{
  "model_name": "Random Forest Regressor",
  "target_variable": "Patient Waittime",
  "mae": 12.34,
  "rmse": 15.67,
  "r2_score": 0.1823
}
```

---

### POST /predict

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Patient Age": 45,
    "Patient Gender": "Female",
    "Patient Race": "White",
    "Department Referral": "General Practice",
    "Patient Admission Flag": "Not Admission"
  }'
```

**Response:**
```json
{
  "predicted_wait_time_minutes": 33.4,
  "urgency_level": "Medium",
  "interpretation": "Predicted wait time is 33.4 minutes (Medium urgency).",
  "input_received": {
    "Patient Age": 45,
    "Patient Gender": "Female",
    "Patient Race": "White",
    "Department Referral": "General Practice",
    "Patient Admission Flag": "Not Admission"
  }
}
```

---

## Model Description

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Regressor |
| n_estimators | 200 |
| max_depth | 10 |
| min_samples_split | 5 |
| min_samples_leaf | 2 |
| Test Split | 20% |
| Cross-validation | 5-fold |
| Preprocessing | OneHotEncoder (categorical) + StandardScaler (numerical) |

### Features Used

| Feature | Type |
|---------|------|
| Patient Age | Numerical |
| Patient Gender | Categorical |
| Patient Race | Categorical |
| Department Referral | Categorical |
| Patient Admission Flag | Categorical |

---

## Evaluation Metrics

> Actual values are generated when the model is trained on the dataset.

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error (minutes) – lower is better |
| RMSE | Root Mean Squared Error (minutes) – lower is better |
| R² | Coefficient of determination – proportion of variance explained |
| CV R² | 5-fold cross-validated R² |
| Baseline MAE | Naive mean-predictor MAE for comparison |

---

## Key Findings

1. **Patient wait times are approximately uniformly distributed** between 0 and 60 minutes, indicating no single bottleneck hour or department
2. **Departments with referrals** (General Practice, Orthopedics, Physiotherapy) tend to have different wait patterns than walk-in patients
3. **Admission patients** generally exhibit higher wait times than non-admitted patients
4. **Senior patients (60+)** have a slightly wider wait time spread compared to younger groups
5. **Patient satisfaction scores** are provided for only ~20% of records; the sparse nature limits correlation analysis
6. **Peak arrival hours** can be identified from the hourly distribution chart, helping with staffing

---

## Future Improvements

1. Incorporate real-time IoT sensor data for live wait-time prediction
2. Add LSTM / time-series models to capture daily/weekly seasonality
3. Integrate Electronic Health Record (EHR) data for richer features
4. Deploy on IBM Cloud / AWS with auto-scaling
5. Build staff scheduling recommendation engine on top of predictions
6. Collect complete satisfaction scores to enable satisfaction prediction
7. Add anomaly detection for unusual patient surges

---

## Single-File Architecture Note

The `SourajitMondal_HospitalPatientFlow.py` file is organised into six clearly labelled sections:

| Section | Lines | Purpose |
|---------|-------|---------|
| SECTION 1 | ~90–160 | Data loading & preprocessing |
| SECTION 2 | ~163–250 | ML model (train, evaluate, predict) |
| SECTION 3 | ~253–360 | Visualisation helpers |
| SECTION 4 | ~363–430 | Flask backend API |
| SECTION 5 | ~433–620 | Streamlit frontend UI |
| SECTION 6 | ~623–650 | Entry-point dispatcher |

When launched via `python SourajitMondal_HospitalPatientFlow.py`, the `main()` function runs. When launched via `streamlit run SourajitMondal_HospitalPatientFlow.py`, the `run_streamlit_ui()` function runs instead.

---

*IBM Bharat / AICTE Data Analytics with AI Virtual Internship — Final Project Submission*
