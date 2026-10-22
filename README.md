# Explainable ML-Based Network Intrusion Detection System (NIDS)

IBM SkillsBuild Cybersecurity Bootcamp 2026 (BharatCares) - Final Project

## Project Description
A machine learning system that detects malicious network connections, classifies the attack type, explains each alert with **SHAP**, and converts predictions into **SOC-style alerts with severity levels and recommended response actions**. A Streamlit dashboard lets you analyse traffic files and test single connections.

**Key features**
- Binary detection (Normal vs Attack) and 5-class classification (Normal, DoS, Probe, R2L, U2R)
- Four models compared with stratified cross-validation: Logistic Regression, Decision Tree, Random Forest, XGBoost
- Leak-free preprocessing inside the scikit-learn pipeline
- Evaluation on the **official KDDTest+ set**, which contains unseen attack types
- Threshold analysis (recall vs false-alarm trade-off) and error analysis by attack category
- SHAP explanations (global summary and a local waterfall for one alert)
- SOAR-style alert log (`alerts_log.csv`) and an interactive Streamlit dashboard

## Dataset
**NSL-KDD** (University of New Brunswick, Canadian Institute for Cybersecurity)
- Kaggle: https://www.kaggle.com/datasets/hassan06/nslkdd
- Official: https://www.unb.ca/cic/datasets/nsl.html
- Files used: `KDDTrain+.txt` (125,973 records) and `KDDTest+.txt` (22,544 records), 41 features each

The notebook downloads the dataset automatically.

## Technologies Used
Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, joblib, Streamlit, Google Colab

## Results (official KDDTest+ set)
> Fill in from Section 15 of the notebook.

| Metric | Value |
|---|---|
| Champion model | FILL_IN |
| Accuracy | FILL_IN |
| Precision | FILL_IN |
| Recall (attacks) | FILL_IN |
| F1-score | FILL_IN |
| ROC-AUC | FILL_IN |

Test scores are lower than cross-validation scores because `KDDTest+` contains attack types that never appear in the training data, which mimics real zero-day behaviour.

## Setup and Run

### 1. Train the model (Google Colab, recommended)
1. Upload `YourName_IntrusionDetector.ipynb` to https://colab.research.google.com
2. Run `Runtime -> Run all` (about 15-20 minutes). The dataset downloads automatically.
3. The notebook creates `model.pkl`, `multiclass_model.pkl`, `scaler.pkl`, `columns.pkl`, `alerts_log.csv`, and the `images/` and `results/` folders.

### 2. Run the dashboard locally
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
pip install -r requirements.txt
streamlit run app.py
```
Place `model.pkl` and `columns.pkl` (from the notebook) in the same folder as `app.py`. Upload `KDDTest+.txt` in the first tab to see the alert dashboard.

## Repository Structure
```
├── YourName_IntrusionDetector.ipynb   # full training and evaluation notebook
├── app.py                             # Streamlit dashboard
├── requirements.txt
├── README.md
├── YourName_ProjectReport.docx
├── model.pkl, multiclass_model.pkl, scaler.pkl, columns.pkl
├── alerts_log.csv
├── images/                            # charts used in the report
└── results/                           # metric tables (CSV)
```

## Limitations
- NSL-KDD is derived from 1999-era traffic; modern threats are not represented.
- Rare classes (R2L, U2R) are hard to detect from flow statistics alone.
- Supervised models struggle with attack types they never saw.

## Future Work
UNSW-NB15 / CIC-IDS2017 evaluation, autoencoder anomaly detection, live packet capture with Scapy or Zeek, threat-intelligence enrichment (VirusTotal, AbuseIPDB).

## Author
Your Name - IBM SkillsBuild Cybersecurity Bootcamp 2026
# ML-Network-Intrusion-Detector
# ML-Network-Intrusion-Detector
# ML-Network-Intrusion-Detector
# ML-Network-Intrusion-Detector
# ML-Network-Intrusion-Detector
# ML-Network-Intrusion-Detector
# ML-Network-Intrusion-Detector
