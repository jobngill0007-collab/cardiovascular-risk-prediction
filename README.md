# ❤️ CardioGuard — Cardiovascular Risk Prediction System

A browser-based AI-powered cardiovascular disease risk assessment tool built as a B.E. final year project at **Chandigarh University** (April 2026).

---

## 👥 Team

| Name | Roll No |
|---|---|
| Sukhvinder Singh | 25BAI70441 |
| Dheirya Goyal | 25BAI70459 |
| Jobanjeet Singh | 25BAI70622 |
| Arunpal Singh Dhaliwal | 25BAI70672 |

**Supervisor:** Dr. Navneet Kaur  
**Degree:** B.E. — Computer Science Engineering (AI & ML)

---

## 🧠 What It Does

Users answer an **18-question self-assessment form** covering lifestyle, medical history, and demographics. The system then provides:

- 🔴 **ML Disease Probability** — from a trained Voting Ensemble model
- 💚 **Heart Health Score** (0–100) — from a clinical scoring engine (Framingham / AHA)
- 📊 **5 Interactive Charts** — probability doughnut, health score ring, risk factor bar, lifestyle radar, model comparison
- 📋 **Personalised Recommendations** — evidence-based lifestyle guidance
- 📥 **Downloadable HTML Health Report**

---

## 🤖 Machine Learning Model

| Property | Detail |
|---|---|
| Algorithm | Voting Ensemble (Random Forest + Gradient Boosting + Logistic Regression) |
| Primary Dataset | CDC BRFSS 2022 (445,132 records) |
| Fallback Dataset | UCI Cleveland Heart Disease |
| CV Accuracy | ~83–88% (10-fold stratified) |
| ROC-AUC | ~0.79–0.80 |
| Overfit Gap | < 5% |
| Tuning | GridSearchCV with anti-overfitting constraints (max_depth=5, min_samples_leaf=5) |

---

## 📁 Project Structure

```
cardiovascular-risk-prediction/
├── heart_health_app.py   # Web server — serves UI on localhost:5050
├── train_model.py        # Model training — saves heart_model.pkl
├── index.html            # Frontend — 18-question form + charts + report
├── heart_model.pkl       # Trained model (generated after training)
└── .gitignore
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install scikit-learn pandas numpy joblib
```

### 2. Add dataset
Place one of the following in the project folder:
- `LLCP2022.XPT` — CDC BRFSS 2022 *(recommended, ~300 MB)*
- `heart.csv` — UCI Cleveland Heart Disease *(smaller, quick test)*

> Download BRFSS 2022 from the [CDC website](https://www.cdc.gov/brfss/annual_data/annual_2022.html)  
> Download UCI dataset from [Kaggle — Heart Disease UCI](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

### 3. Train the model
```bash
python train_model.py
```
Generates `heart_model.pkl` with all metrics.

### 4. Launch the app
```bash
python heart_health_app.py
```
Browser opens automatically at **http://localhost:5050**

---

## 🖥️ Screenshots

> *(Add screenshots of the UI here)*

---

## 📚 References

- World Health Organization — Cardiovascular Diseases Fact Sheet
- CDC — Behavioral Risk Factor Surveillance System (BRFSS) 2022
- D'Agostino RB et al. — General Cardiovascular Risk Profile. *Circulation*, 2008
- Yusuf S et al. — INTERHEART Study. *The Lancet*, 2004
- Lloyd-Jones DM et al. — Life's Essential 8. *Circulation*, 2022
- Pedregosa F et al. — Scikit-learn: Machine Learning in Python. *JMLR*, 2011

---

## ⚠️ Disclaimer

This tool is for **educational awareness only** and does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.
