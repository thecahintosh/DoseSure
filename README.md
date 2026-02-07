# 🧬 Antibiotic Resistance Prediction System  
**Health & Well-Being | Clinical Decision Support**

A machine learning system that predicts **antibiotic resistance probabilities** for bacterial cultures to support safer, more effective empiric antibiotic selection.

This project addresses a critical public health challenge: **antimicrobial resistance (AMR)** — one of the top global threats to health and well-being.

---

## 🌍 Problem Statement

Clinicians often prescribe empiric antibiotics **before susceptibility results are available**, relying on population-level antibiograms that:

- Ignore organism-specific patterns  
- Ignore specimen type  
- Ignore temporal trends  
- Increase the risk of treatment failure and resistance  

This can lead to:
- Delayed effective therapy
- Unnecessary broad-spectrum antibiotic use
- Worsening antimicrobial resistance

---

## 💡 Solution Overview

This system predicts **culture-specific resistance risk** for 15 commonly used antibiotics using large-scale microbiology data and machine learning.

Instead of asking:
> “What usually works?”

We ask:
> **“What is most likely to work for *this* culture?”**

---

## 📊 Dataset

**ARMD-MGB (de-identified clinical microbiology dataset)**

- ~5 million microbiology records  
- ~200,000 unique cultures  
- Multiple specimen types (urine, blood, respiratory)  
- Longitudinal data spanning decades  
- Fully de-identified patient information  

---

## 🏗️ System Architecture

### 1. Data Processing
- Automatic ingestion of all microbiology tables
- Standardization of susceptibility labels
- Filtering to clinically relevant antibiotics
- Culture-level aggregation

### 2. Feature Engineering
- **Organism features**
  - Top 30 organisms (covers ~97% of cultures)
  - Gram-positive / gram-negative classification
- **Specimen features**
  - Urine, blood, respiratory
- **Temporal features**
  - Year, month, weekday, hour
  - Weekend vs weekday
  - Day vs night
- **Population antibiogram baseline**

**Total features:** 44

---

## 🤖 Modeling Approach

- **Model type:** XGBoost (binary classification)
- **One model per antibiotic** (15 total)
- **Temporal train/test split** to prevent data leakage
- Class imbalance handled using weighted loss

### Target Antibiotics
- Ciprofloxacin
- Levofloxacin
- Trimethoprim–Sulfamethoxazole
- Nitrofurantoin
- Ceftriaxone
- Cefazolin
- Ampicillin
- Ampicillin–Sulbactam
- Piperacillin–Tazobactam
- Gentamicin
- Cefepime
- Meropenem
- Aztreonam
- Tobramycin
- Amikacin

---

## 📈 Key Performance Metrics

All metrics reported on a **held-out temporal test set**.

- **Mean AUROC:** ~0.73   
- **Average AUPRC lift:** ~4× over random  
- Best-performing models:
  - **Amikacin:** AUROC ≈ 0.94
  - **Meropenem:** AUROC ≈ 0.91
  - **Nitrofurantoin:** AUROC ≈ 0.89

Performance consistently exceeds population antibiogram baselines.

---

## ⚠️ Clinical Safety Analysis

Special emphasis was placed on **false negatives**  
(predicted susceptible but actually resistant — the most dangerous error).

Findings:
- Lower decision thresholds (0.2–0.3) significantly reduce false negatives
- Improves sensitivity at the cost of some false positives
- Aligns with real-world clinical safety priorities

---

## 🖥️ Interactive Gradio Demo

A **Gradio web application** allows users to:

1. Select:
   - Organism
   - Specimen type
   - Temporal context
2. Generate:
   - Resistance probability per antibiotic
   - Ranked antibiotic recommendations
3. Compare:
   - Model predictions vs population antibiogram

This makes the system **immediately usable without reading the code**.

---

## ▶️ How to Run
### Download all core files of the dataset using download_armdmgb.py

### Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn gradio
```
### Run every cell, last cell will give you necessary external url to run the UI.
