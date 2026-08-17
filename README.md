<div align="center">

# ⚡ Electrochemical ML Analysis

**Machine learning models for reproducing cyclic voltammetry (CV) current response in TiO₂–MnO₂-based supercapacitor electrodes**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](#-google-colab)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](#)

</div>

---

## 📋 Overview

This repository benchmarks **14 regression models** to model and reproduce experimentally measured cyclic voltammetry current responses for pristine and metal-doped (Ag, Pb, Bi) TiO₂–MnO₂ electrode systems.

> ⚠️ **Scope note:** Models are trained/evaluated using point-wise train/test splits drawn from *within* each individual CV curve. Reported metrics (R², RMSE, MAE, MSE) reflect **within-dataset interpolation performance**, not generalized prediction for unseen materials, dopants, or independently measured replicate curves. See [Limitations](#-limitations).

---

## ✨ Features

| | |
|---|---|
| 🧠 **14 ML models** | RF, GB, Extra Trees, HistGB, SVR, KNN (k=5,7,9), MLP, Linear, XGBoost, LightGBM, CatBoost |
| 🏆 **Stacking Regressor** | RF + GB + SVR base learners → Linear Regression meta-learner — best performer across all systems |
| ⚡ **Scan-rate resolved** | 5, 10, 15, 20, 25, 50 mV s⁻¹ analyzed independently |
| 🔁 **Fully reproducible** | Single global seed (`random_state = 42`) applied to every split and every stochastic model |
| 📊 **Auto-generated outputs** | Scatter plots, comparison charts, boxplots, metrics tables, config logs |
| 📦 **One-click packaging** | All results zipped and auto-downloaded (Colab-friendly) |

---

## 📁 Repository Structure

```
├── src/
│   └── main.py                          # main analysis script
├── requirements.txt
├── README.md
└── outputs/                             # generated at runtime
    ├── <Material>/
    │   ├── scatter_combined.png
    │   ├── R2_comparison.png  RMSE_comparison.png  MAE_comparison.png  MSE_comparison.png
    │   ├── box_R2.png  box_RMSE.png  box_MAE.png  box_MSE.png
    │   └── all_model_metrics.xlsx
    ├── Stacking_R2.png  Stacking_RMSE.png  Stacking_MAE.png  Stacking_MSE.png
    ├── dataset_sizes_per_material_scan_rate.xlsx
    ├── model_hyperparameters_and_seeds.json
    └── ml_analysis_results.zip          # everything above, zipped
```

---

## 🚀 Getting Started

### 💻 Local

```bash
pip install -r requirements.txt
```

Place your input Excel files (potential/current column pairs per scan rate — one file per material composition) in the working directory, set `file_names` in `src/main.py`, then run:

```bash
python src/main.py
```

### 🔬 Google Colab

```python
!pip install catboost lightgbm -q
```

1. Upload your Excel files when prompted, or mount Google Drive and point `file_names` to the file paths
2. Run the analysis cells
3. Run the final cell to bundle everything into `ml_analysis_results.zip` and trigger an automatic download

---

## 📥 Input Data Format

Each Excel file corresponds to one electrode composition —
`TiO2-MnO2.xlsx` · `Ag-TiO2-MnO2.xlsx` · `Pb-TiO2-MnO2.xlsx` · `Bi-TiO2-MnO2.xlsx`

— and contains alternating **potential/current** column pairs, one pair per scan rate (5, 10, 15, 20, 25, 50 mV s⁻¹). Rows with missing values are dropped before modeling.

> Sweep direction (anodic/cathodic) and time-step index are **not** currently included as separate columns — see [Limitations](#-limitations).

---

## 🔁 Reproducibility

- ✅ Single global seed `random_state = 42` applied to `train_test_split` **and** every stochastic model (RF, GB, Extra Trees, HistGB, MLP, XGBoost, LightGBM, CatBoost, plus the RF/GB base learners inside the Stacking Regressor)
- ✅ Random Forest hyperparameters harmonized (`n_estimators = 200`) everywhere, including inside the Stacking Regressor
- ✅ Key hyperparameters: MLP `max_iter = 2000` · KNN `k = 5, 7, 9` · all else at library defaults
- ✅ Exact configuration auto-logged to `model_hyperparameters_and_seeds.json` every run
- ✅ Per-material, per-scan-rate sample counts auto-logged to `dataset_sizes_per_material_scan_rate.xlsx` every run

---

## 📤 Output

- 🔵 Scatter plots — actual vs. modeled current response (Stacking Regressor), per material & scan rate
- 📈 Model-comparison bar / line / box plots (MSE, RMSE, MAE, R²) across all 14 models
- 📑 Per-material Excel file with full metrics for every model × scan rate
- 🔢 Dataset-size table (training/test sample counts)
- ⚙️ Hyperparameter & random-seed configuration log (JSON)
- 🗜️ All of the above bundled into one downloadable `.zip`

---

## ⚠️ Limitations

- **Input features** — Only applied potential (V) is used. Sweep direction and time-step index aren't included, so current at a given potential is effectively averaged across the anodic and cathodic branches rather than resolved separately.
- **Validation strategy** — Train/test splits are point-wise, within each individual CV curve (80:20), not across independent replicate curves. Metrics reflect within-dataset interpolation, not generalized prediction on unseen data.
- **No cross-validation, hyperparameter tuning, or uncertainty quantification** (e.g., mean ± SD over repeated runs) is currently performed.

**Planned improvements:** sweep-direction / time-index features · replicate-curve validation · grouped/k-fold cross-validation · systematic hyperparameter optimization · uncertainty quantification · material descriptors (dopant type, ionic radius, dopant charge, structural parameters) for generalized materials-level modeling.

---

## 👤 Author

**Safi Ullah Majid**
