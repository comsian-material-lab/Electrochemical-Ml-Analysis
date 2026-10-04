<div align="center">

# ⚡ Electrochemical ML Analysis

**Machine learning models for reproducing cyclic voltammetry (CV) current response in TiO₂–MnO₂-based supercapacitor electrodes, with leakage-controlled validation**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](#-google-colab)
[![Release](https://img.shields.io/badge/release-v1.3-brightgreen.svg)](#)

</div>

---

## 📋 Overview

This repository compares regression models for reproducing experimentally measured cyclic voltammetry current responses of pristine and metal-doped (Ag, Pb, Bi) TiO₂–MnO₂ electrodes. Inputs are applied potential, a sweep-direction indicator and a scan-rate index (plus seven dopant descriptors). Performance is judged with five validation schemes, from random splits within one curve to leave-one-scan-rate-out and block cross-validation.

> ⚠️ **Scope note:** The models reproduce the measured electrodes at held-out potential windows and scan rates. Replicate CV curves were not measured, so the results do **not** show prediction for independent experiments, new dopants or new materials. See [Limitations](#-limitations).

---

## 🆕 What's new in v1.3

- **Sweep-direction and scan-rate inputs.** The six scan rates of each electrode are modelled together in one dataset (previously one model per scan rate, potential only).
- **Five validation schemes** for Extra Trees and HistGB (see below).
- **Purged 5-fold CV** for model comparison: Extra Trees, Random Forest, HistGB, LightGBM and the Stacking Regressor.
- **Feature ablation, nested hyperparameter tuning, split-conformal prediction intervals, leave-one-electrode-out test of the descriptors, SHAP.**
- **Extra Trees**, not the Stacking Regressor, is the best model under the stricter validation.
- Dataset and fold sizes, and sweep-label checks, are logged to the Excel report.

---

## 🧪 Validation schemes

| # | Scheme | What it tests |
|---|---|---|
| 1 | Holdout 80:20, per scan-rate curve | Interpolation within one curve (leaky) |
| 2 | Shuffled 5-fold, per curve | Same, repeated (leaky) |
| 3 | Block 5-fold, per curve (no shuffling) | Extrapolation to unseen segments of a curve |
| 4 | Leave-one-scan-rate-out | A complete curve at an unseen scan rate |
| 5 | Combined model: 5a shuffled, 5b **purged** 5-fold | Whole potential windows held out; a 3-band buffer around each test band is removed from training |

Purged CV splits the potential axis into 200 bands and uses 40 random bands per test fold, so each training set holds about 19–25 % of the points (see `Fold_Sizes_Pooled`).

---

## 📊 Main results (Extra Trees, R²)

| Electrode | Purged 5-fold | Leave-one-scan-rate-out | Block 5-fold (pooled R²) | Random splits |
|---|---|---|---|---|
| Ag-TiO₂-MnO₂ | 0.956 ± 0.031 | 0.962 ± 0.057 | 0.82 | 1.00 |
| Bi-TiO₂-MnO₂ | 0.880 ± 0.086 | 0.853 ± 0.278 | 0.00 | 1.00 |
| Pb-TiO₂-MnO₂ | 0.951 ± 0.036 | 0.961 ± 0.080 | 0.73 | 0.99 |
| TiO₂-MnO₂ | 0.963 ± 0.033 | 0.929 ± 0.169 | 0.67 | 1.00 |

- Potential alone gives R² = 0.18–0.50 under purged CV; sweep direction and scan rate are needed for 0.88–0.96.
- Random splits within a curve give R² ≈ 1.00 and mainly reflect interpolation.
- Extra Trees > Random Forest > Stacking ≈ HistGB ≈ LightGBM on every electrode.
- The hardest held-out scan rates are the extremes (5 and 50 mV s⁻¹).
- The material descriptors are constant within each electrode and add nothing; Bayesian tuning of HistGB did not help.

---

## 📁 Repository Structure

```
├── src/
│   ├── ultimate_analysis_v2.py          # full analysis (all schemes, tables, figures)
│   └── make_figures.py                  # rebuilds figures from ML_REPORT_v2.xlsx
├── requirements.txt
├── README.md
├── ML_REPORT_v2.xlsx                    # results of the paper run
├── Supplementary_Information.docx
└── ULTIMATE_RESULTS/                    # generated at runtime
    ├── ML_REPORT_v2.xlsx                # all result tables
    ├── <Material>_scatter_oof.png       # out-of-fold predicted vs experimental
    ├── <Material>_purged_cv_r2.png      # model comparison under purged CV
    ├── improvement_comparison.png       # feature ablation
    ├── validation_schemes_comparison.png
    ├── purged_vs_standard_cv.png
    ├── shap_importance.png
    └── sweep_direction_check.png
```

---

## 🚀 Getting Started

### 💻 Local

```bash
pip install -r requirements.txt
```

Place the four input Excel files (`TiO2-MnO2.xlsx`, `Ag-TiO2-MnO2.xlsx`, `Pb-TiO2-MnO2.xlsx`, `Bi-TiO2-MnO2.xlsx`) in the working directory, then run:

```bash
python src/ultimate_analysis_v2.py     # full analysis
python src/make_figures.py             # optional: rebuild figures from the report
```

Settings are at the top of `ultimate_analysis_v2.py`:

| Setting | Default | Meaning |
|---|---|---|
| `NESTED_TUNING` | `True` | Bayesian tuning of HistGB inside each purged fold (slow; needs `scikit-optimize`) |
| `RUN_STANDARD_CV` | `True` | Shuffled 5-fold for all 14 models (optimistic, for comparison only) |
| `USE_DESCRIPTORS` | `True` | `False` uses the 3 experimental inputs only |
| `SWEEP_WINDOW` | `21` | Majority-vote window for sweep labels; raise (e.g. 51) if labels are noisy |
| `ONLY_PER_CURVE` | `False` | `True` runs schemes 1–3 only (fast) |

The full run is long (SHAP and nested tuning dominate); set `NESTED_TUNING = False` to shorten it.

### 🔬 Google Colab

1. Upload the script, then run it in a cell with `%run ultimate_analysis_v2.py`.
2. Upload the four Excel files when prompted.
3. Download `ULTIMATE_RESULTS/` (zip it first with `!zip -r results.zip ULTIMATE_RESULTS`).

---

## 📥 Input Data Format

Each Excel file is one electrode composition and has potential/current column pairs per scan rate:
`Potential_5` / `5 mVsec-1`, `Potential_10` / `10 mVsec-1`, …, `Potential_50` / `50 mVsec-1`. Rows with missing values are dropped.

The sweep direction is **inferred** from the order of the rows: 1 where the potential increases, 0 where it decreases; repeated potentials inherit the previous direction, and a 21-point majority vote removes isolated noisy flips. It is not a recorded time axis. Check `sweep_direction_check.png` and the `Sweep_Direction_Check` sheet for your own data.

---

## 🔁 Reproducibility

- Single seed `random_state = 42` for every model, split and calibration step.
- Random Forest and Extra Trees: `n_estimators = 200`. MLP: `max_iter = 2000`. KNN: `k = 5, 7, 9`.
- Stacking Regressor: HistGB, KNN (k = 7) and LightGBM base learners, Ridge (α = 1) meta-learner, internal 5-fold CV.
- SVR, KNN and MLP inputs are standardised inside each training fold.
- All other hyperparameters are library defaults.
- Dataset and fold sizes are logged to the Excel report (`Dataset_Sizes`, `Fold_Sizes_Pooled`, `Fold_Sizes_PerCurve`).

---

## 📤 Output

`ML_REPORT_v2.xlsx` contains: `Validation_Summary`, `Schemes1-3_Pooled_NRMSE`, `Per_Curve_Detail_Schemes1-3`, `Scheme4_Leave1ScanRateOut`, `Purged_5Fold_Models`, `Feature_Ablation`, `Uncertainty`, `Leave1Electrode_Out`, `Fold_Sizes_Pooled`, `Fold_Sizes_PerCurve`, `Dataset_Sizes`, `Sweep_Direction_Check`, `Standard_5Fold_14models`, `SHAP_Importance`, `Feature_List`.

---

## ⚠️ Limitations

- **No replicate curves.** Replicate-based validation was not possible; leave-one-scan-rate-out is the closest available check and is not equivalent to replicates.
- **No generalisation claim.** The models are not shown to predict independent experiments or new materials. The seven descriptors are constant within each electrode, and with four materials a pooled leave-one-electrode-out test gave inconsistent results.
- **Limited extrapolation.** Block CV gives R² = 0.00–0.82 (0.00 for Bi), and the lowest and highest scan rates are the hardest held-out curves.
- **Inferred sweep direction.** Labels come from the data order, not from a recorded time axis.
- **Prediction intervals under-cover.** Split-conformal 90 % intervals covered 60–72 % of held-out points.
- **Model choice.** Extra Trees was identified on the same folds used for reporting, so its scores may be slightly optimistic. Models other than the five compared under purged CV were evaluated with shuffled CV only.
- **Bi-TiO₂-MnO₂** is the weakest and most variable system.

**Planned work:** replicate CV curves as independent test sets, recorded sweep direction or time index, more dopants and compositions, calibrated uncertainty estimates.

---

## 📝 Citation

If you use this code, please cite the associated article and this release (Zenodo DOI: _to be added_).

---

## 👤 Author

**Safi Ullah Majid**
