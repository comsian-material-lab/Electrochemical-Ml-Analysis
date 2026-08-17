# Electrochemical ML Analysis

This repository contains machine learning models used to **model and reproduce (interpolate)** the experimentally measured cyclic voltammetry (CV) current response of TiO₂–MnO₂-based electrode systems (pristine, Ag-, Pb-, and Bi-doped).

> **Note on scope:** These models are trained and evaluated using point-wise train/test splits drawn from within each individual CV curve. As such, the reported metrics (R², RMSE, MAE, MSE) reflect **within-dataset interpolation performance**, not generalized prediction for unseen materials, dopants, or independently measured replicate curves. See *Limitations* below.

## Features
- 14 regression models benchmarked per material and scan rate: Random Forest, Gradient Boosting, Extra Trees, HistGB, SVR, KNN (k = 5, 7, 9), MLP (ANN), Linear Regression, XGBoost, LightGBM, CatBoost, and a Stacking Regressor (RF + GB + SVR base learners, Linear Regression meta-learner)
- Stacking Regressor consistently achieves the highest within-dataset agreement across all electrode systems
- Scan-rate-resolved analysis (5, 10, 15, 20, 25, 50 mV s⁻¹)
- Fully reproducible: a single global random seed (`random_state = 42`) is applied to every train/test split and every stochastic model component
- Automatic generation of scatter plots, model-comparison plots, boxplots, a per-material metrics table, a dataset-size table, and a hyperparameter/configuration log
- One-step results compression to a downloadable `.zip` (Colab-friendly)

## Repository Structure
```
├── src/
│   └── main.py                  # main analysis script
├── requirements.txt
├── README.md
└── outputs/                     # generated at runtime (not tracked in repo)
    ├── <Material>/
    │   ├── scatter_combined.png
    │   ├── R2_comparison.png / RMSE_comparison.png / MAE_comparison.png / MSE_comparison.png
    │   ├── box_R2.png / box_RMSE.png / box_MAE.png / box_MSE.png
    │   └── all_model_metrics.xlsx
    ├── Stacking_R2.png / Stacking_RMSE.png / Stacking_MAE.png / Stacking_MSE.png
    ├── dataset_sizes_per_material_scan_rate.xlsx
    ├── model_hyperparameters_and_seeds.json
    └── ml_analysis_results.zip   # all of the above, zipped for download
```

## How to Run

### Locally
1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Place your input Excel files (potential/current column pairs per scan rate, one file per material composition) in the working directory, and set `file_names` in `src/main.py` accordingly.
3. Run the script:
   ```
   python src/main.py
   ```

### Google Colab
1. Install the packages not pre-installed in Colab:
   ```python
   !pip install catboost lightgbm -q
   ```
2. Upload your Excel files when prompted, or mount Google Drive and point `file_names` to the file paths.
3. Run the analysis cells.
4. Run the final compression cell to bundle all outputs into a single `ml_analysis_results.zip` and trigger an automatic browser download.

## Input Data Format
Each Excel file corresponds to one electrode composition (e.g., `TiO2-MnO2.xlsx`, `Ag-TiO2-MnO2.xlsx`, `Pb-TiO2-MnO2.xlsx`, `Bi-TiO2-MnO2.xlsx`) and contains alternating potential/current column pairs, one pair per scan rate (5, 10, 15, 20, 25, 50 mV s⁻¹). Rows with missing values are dropped before modeling. **Sweep direction (anodic/cathodic) and time-step index are not currently included as separate columns**; see *Limitations*.

## Reproducibility
- A single global seed, `random_state = 42`, is applied to `train_test_split` and to every stochastic model (Random Forest, Gradient Boosting, Extra Trees, HistGB, MLP, XGBoost, LightGBM, CatBoost, and the Random Forest/Gradient Boosting base learners inside the Stacking Regressor).
- Random Forest hyperparameters are harmonized (`n_estimators = 200`) across all uses, including inside the Stacking Regressor.
- Key hyperparameters: MLP `max_iter = 2000`; KNN `k = 5, 7, 9`; all other parameters left at library defaults (scikit-learn / XGBoost / LightGBM / CatBoost).
- Exact model configuration is logged automatically to `model_hyperparameters_and_seeds.json` on every run.
- Per-material, per-scan-rate training/test sample counts are logged automatically to `dataset_sizes_per_material_scan_rate.xlsx` on every run.

## Output
- Scatter plots of actual vs. modeled current response (Stacking Regressor), per material, per scan rate
- Model-comparison bar/line/box plots (MSE, RMSE, MAE, R²) across all 14 models
- Per-material Excel file with full metrics for every model × scan rate combination
- Dataset-size table (training/test sample counts per material and scan rate)
- Hyperparameter and random-seed configuration log (JSON)
- All of the above bundled into a single downloadable `.zip`

## Limitations
- **Input features:** Only applied potential (V) is used as the model input. Sweep direction and time-step index are not included, so current values at a given potential are effectively averaged across the anodic and cathodic branches of the CV cycle rather than resolved separately.
- **Validation strategy:** Train/test splits are performed point-wise within each individual CV curve (80:20 split), not across independent replicate curves. Reported metrics therefore reflect within-dataset interpolation performance rather than generalized prediction for unseen or independently measured data.
- **No cross-validation, hyperparameter tuning, or uncertainty quantification** (e.g., mean ± SD across repeated runs) is currently performed.

Planned/recommended improvements: incorporate sweep-direction or a sequential time-index feature; retain and use complete replicate CV curves as independent test sets; add grouped/k-fold cross-validation, systematic hyperparameter optimization, and uncertainty quantification; extend inputs with material descriptors (dopant type, ionic radius, dopant charge, structural parameters) for more generalized materials-level modeling.

## Author
Safi Ullah Majid
