# Explainable ML-Based Network Intrusion Detection System (NIDS)

**IBM SkillsBuild Cybersecurity Bootcamp 2026 (BharatCares) - Final Project**
Author: Sakshi vinod mankar | Bsc forensic science | sandip university nashik

## Project Description
A machine-learning system that detects malicious network connections, classifies the attack type, explains every alert
with **SHAP**, and turns predictions into **SOC-style alerts with severity levels and recommended actions**.
A Streamlit dashboard lets you analyse traffic files, simulate live traffic and test single connections.

**Key features**
- Binary detection (Normal vs Attack) and 5-class classification (Normal, DoS, Probe, R2L, U2R)
- Four models compared with stratified cross-validation: Logistic Regression, Decision Tree, Random Forest, XGBoost
- Leak-free preprocessing inside the scikit-learn pipeline
- Evaluation on the **official KDDTest+ set**, which contains attack types unseen in training
- Threshold analysis, error analysis, SHAP explanations and a SOAR-style alert log
- Interactive dashboard (CyberShield NIDS)

## Dataset
**NSL-KDD** - Canadian Institute for Cybersecurity, University of New Brunswick
- Kaggle: https://www.kaggle.com/datasets/hassan06/nslkdd
- Official page: https://www.unb.ca/cic/datasets/nsl.html
- Files: `KDDTrain+.txt` (125,973 records) and `KDDTest+.txt` (22,544 records), 41 features per connection

The notebook downloads the dataset automatically.

## Technologies Used
Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Altair, joblib, Streamlit, Google Colab

## Results (official KDDTest+ set)
Champion model: **XGBoost (Tuned)**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7540 | 0.9168 | 0.6246 | 0.7430 | 0.7904 |
| Decision Tree | 0.7976 | 0.9654 | 0.6684 | 0.7899 | 0.8183 |
| Random Forest (Tuned) | 0.7698 | 0.9676 | 0.6162 | 0.7529 | 0.9686 |
| XGBoost (Tuned) | 0.8031 | 0.9690 | 0.6757 | 0.7962 | 0.9723 |

Test scores are lower than cross-validation scores because KDDTest+ contains attack types that never appear in training,
which mimics real zero-day behaviour.

## Setup and Run

### 1. Train the model (Google Colab, recommended)
1. Open `SakshiMankar_IntrusionDetector.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run `Runtime -> Run all` (about 15-20 minutes). The dataset is downloaded automatically.
3. The notebook creates `model.pkl`, `multiclass_model.pkl`, `scaler.pkl`, `columns.pkl`, `alerts_log.csv`, `images/` and `results/`.

### 2. Run the dashboard
```bash
git clone https://github.com/sakshimankar23/ML-Network-Intrusion-Detector.git.git
cd ML-Network-Intrusion-Detector.git
pip install -r requirements.txt
streamlit run app.py
```
Use the **same library versions as Colab** (scikit-learn and XGBoost) so that `model.pkl` loads correctly, or run the app
inside Colab. For demo data, place `KDDTest+.txt` in a folder named `data/`.

## Repository Structure
```
SakshiMankar_IntrusionDetector.ipynb
app.py
requirements.txt
README.md
Sakshi_vinod_mankar_ProjectReport.docx
model.pkl, multiclass_model.pkl, scaler.pkl, columns.pkl
alerts_log.csv
images/        charts used in the report
results/       metric tables (CSV)
screenshots/   dashboard screenshots
```

## Key Information
- Severity levels: Low (below threshold), Medium (up to 70%), High (70-90%), Critical (above 90%)
- Weakest classes: R2L and U2R, because they resemble normal user behaviour and are rare in the training data
- Limitations: NSL-KDD is an older benchmark; real networks are noisier; unseen attacks are hard for supervised models
- Future work: UNSW-NB15 / CIC-IDS2017, autoencoder anomaly detection, live packet capture, threat-intelligence enrichment

## Author
Sakshi vinod mankar - IBM SkillsBuild Cybersecurity Bootcamp 2026
