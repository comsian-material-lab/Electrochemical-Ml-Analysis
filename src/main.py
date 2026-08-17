import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, HistGradientBoostingRegressor,
    StackingRegressor
)

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ==============================
# GLOBAL REPRODUCIBILITY SETTINGS
# ==============================
# A single, uniform random seed is used everywhere a stochastic process occurs:
# train/test splitting AND every model that has internal randomness
# (tree bootstrapping, weight initialization, subsampling, etc.).
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==============================
# COLORS
# ==============================
def get_14_distinct_colors():
    return [
        "#e41a1c","#377eb8","#4daf4a","#984ea3",
        "#ff7f00","#ffff33","#a65628","#f781bf",
        "#999999","#66c2a5","#fc8d62","#8da0cb",
        "#e78ac3","#a6d854"
    ]

# ==============================
# SUBSCRIPT FUNCTION
# ==============================
def to_subscript(text):
    sub_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return text.translate(sub_map)

# ==============================
# MODELS
# ==============================
# NOTE ON HYPERPARAMETERS:
# Random Forest is fixed at n_estimators=200 everywhere it appears, including
# as a base learner inside the Stacking Regressor (previously this was
# inconsistently set to 150 there). random_state=42 is now applied to every
# stochastic estimator so that results are exactly reproducible across runs.
RF_N_ESTIMATORS = 200
ET_N_ESTIMATORS = 200
MLP_MAX_ITER = 2000
KNN_KS = [5, 7, 9]

def get_models():
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=ET_N_ESTIMATORS, random_state=RANDOM_STATE
        ),
        "HistGB": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
        "SVR": SVR(),  # SVR (RBF kernel) is deterministic given fixed data/scaling; no random_state param exists
        "KNN (5)": KNeighborsRegressor(KNN_KS[0]),
        "KNN (7)": KNeighborsRegressor(KNN_KS[1]),
        "KNN (9)": KNeighborsRegressor(KNN_KS[2]),
        "MLP": MLPRegressor(
            max_iter=MLP_MAX_ITER, random_state=RANDOM_STATE
        ),
        "Linear": LinearRegression(),  # deterministic, no seed needed
        "XGBoost": XGBRegressor(
            random_state=RANDOM_STATE
        ),
        "LightGBM": LGBMRegressor(
            random_state=RANDOM_STATE
        ),
        "CatBoost": CatBoostRegressor(
            verbose=0, random_state=RANDOM_STATE
        )
    }

    stacking = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE
            )),
            ("gbr", GradientBoostingRegressor(
                random_state=RANDOM_STATE
            )),
            ("svr", SVR())
        ],
        final_estimator=LinearRegression()
    )

    models["Stacking"] = stacking
    return models

# ==============================
# HYPERPARAMETER / CONFIG LOG (for Supplementary Table)
# ==============================
def log_model_config():
    """
    Records the exact hyperparameters and random seed used for every model,
    to be reported in the manuscript / supplementary materials for full
    reproducibility, as requested by the reviewer.
    """
    config = {
        "global_random_state": RANDOM_STATE,
        "train_test_split": {"test_size": 0.2, "random_state": RANDOM_STATE},
        "models": {
            "Random Forest": {"n_estimators": RF_N_ESTIMATORS, "random_state": RANDOM_STATE,
                               "other_params": "scikit-learn defaults"},
            "Gradient Boosting": {"random_state": RANDOM_STATE, "other_params": "scikit-learn defaults"},
            "Extra Trees": {"n_estimators": ET_N_ESTIMATORS, "random_state": RANDOM_STATE,
                             "other_params": "scikit-learn defaults"},
            "HistGB": {"random_state": RANDOM_STATE, "other_params": "scikit-learn defaults"},
            "SVR": {"kernel": "rbf", "other_params": "scikit-learn defaults (deterministic, no seed)"},
            "KNN (5)": {"n_neighbors": 5, "other_params": "scikit-learn defaults (deterministic, no seed)"},
            "KNN (7)": {"n_neighbors": 7, "other_params": "scikit-learn defaults (deterministic, no seed)"},
            "KNN (9)": {"n_neighbors": 9, "other_params": "scikit-learn defaults (deterministic, no seed)"},
            "MLP": {"max_iter": MLP_MAX_ITER, "random_state": RANDOM_STATE,
                     "other_params": "scikit-learn defaults"},
            "Linear": {"other_params": "scikit-learn defaults (deterministic, no seed)"},
            "XGBoost": {"random_state": RANDOM_STATE, "other_params": "xgboost defaults"},
            "LightGBM": {"random_state": RANDOM_STATE, "other_params": "lightgbm defaults"},
            "CatBoost": {"random_state": RANDOM_STATE, "verbose": 0, "other_params": "catboost defaults"},
            "Stacking Regressor": {
                "base_learners": [
                    {"name": "Random Forest", "n_estimators": RF_N_ESTIMATORS, "random_state": RANDOM_STATE},
                    {"name": "Gradient Boosting", "random_state": RANDOM_STATE},
                    {"name": "SVR", "kernel": "rbf"}
                ],
                "final_estimator": "Linear Regression"
            }
        }
    }
    with open("model_hyperparameters_and_seeds.json", "w") as f:
        json.dump(config, f, indent=2)
    return config

# ==============================
# STORAGE
# ==============================
stacking_summary = {}
dataset_size_records = []  # collects per-material, per-scan-rate dataset sizes

# ==============================
# MAIN LOOP
# ==============================
log_model_config()

for file in file_names:

    print(f"\n🚀 Processing: {file}")

    file_clean = file.replace(".xlsx","").replace(".xls","")
    os.makedirs(file_clean, exist_ok=True)

    data = pd.read_excel(file)
    columns = data.columns

    models = get_models()
    results = {m: {} for m in models}
    scan_rates = []

    # ==============================
    # TRAIN MODELS
    # ==============================
    for i in range(0, len(columns), 2):

        potential_col = columns[i]
        current_col = columns[i+1]

        scan_rate = str(current_col)
        scan_rates.append(scan_rate)

        df = data[[potential_col, current_col]].dropna()

        X = df[[potential_col]].values
        y = df[current_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        # Record dataset sizes for the Supplementary Table requested by the reviewer
        dataset_size_records.append({
            "Material": file_clean,
            "Scan Rate": scan_rate,
            "Total Points (after dropna)": len(df),
            "Training Samples": len(X_train),
            "Test Samples": len(X_test)
        })

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for name, model in models.items():
            try:
                if name in ["SVR","MLP","KNN (5)","KNN (7)","KNN (9)"]:
                    model.fit(X_train_s, y_train)
                    y_pred = model.predict(X_test_s)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                results[name][scan_rate] = {
                    "MSE": mse,
                    "RMSE": np.sqrt(mse),
                    "MAE": mae,
                    "R2": r2_score(y_test, y_pred)
                }

            except Exception as e:
                print(f"⚠️  {name} failed on {file_clean} / {scan_rate}: {e}")
                results[name][scan_rate] = None

    # ==============================
    # SCATTER (STACKING)
    # ==============================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, sr in enumerate(scan_rates):

        potential_col = columns[idx*2]
        current_col = columns[idx*2 + 1]

        df = data[[potential_col, current_col]].dropna()

        X = df[[potential_col]].values
        y = df[current_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        stack_model = StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(
                    n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE
                )),
                ("gbr", GradientBoostingRegressor(
                    random_state=RANDOM_STATE
                )),
                ("svr", SVR())
            ],
            final_estimator=LinearRegression()
        )

        stack_model.fit(X_train, y_train)
        y_pred = stack_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        ax = axes[idx]
        ax.scatter(X_test, y_test, label="Actual", alpha=0.7)
        ax.scatter(X_test, y_pred, label="Predicted", alpha=0.7)

        ax.set_title(f"MSE: {mse:.4f}, R²: {r2:.4f}")
        ax.set_xlabel("Potential (V)")
        ax.set_ylabel("Current")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{file_clean}/scatter_combined.png", dpi=300)
    plt.close()

    # ==============================
    # STACKING STORAGE
    # ==============================
    stacking_summary[file_clean] = {"R2":[],"RMSE":[],"MAE":[],"MSE":[]}

    for sr in scan_rates:
        res = results["Stacking"][sr]
        if res is not None:
            stacking_summary[file_clean]["R2"].append(res["R2"])
            stacking_summary[file_clean]["RMSE"].append(res["RMSE"])
            stacking_summary[file_clean]["MAE"].append(res["MAE"])
            stacking_summary[file_clean]["MSE"].append(res["MSE"])

    # ==============================
    # LINE GRAPHS
    # ==============================
    for metric in ["R2","RMSE","MSE","MAE"]:

        plt.figure(figsize=(10,6))
        colors = get_14_distinct_colors()

        for idx, model in enumerate(models):
            vals = [results[model][sr][metric]
                    for sr in scan_rates if results[model][sr] is not None]

            plt.plot(scan_rates[:len(vals)], vals,
                     marker='o', linewidth=2.5,
                     color=colors[idx], label=model)

        if metric == "R2":
            plt.title(r"$R^2$ Comparison")
            plt.ylabel(r"$R^2$")
        else:
            plt.title(f"{metric} Comparison")
            plt.ylabel(metric)

        plt.xlabel("Scan Rate")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(fontsize=7, ncol=2)

        plt.savefig(f"{file_clean}/{metric}_comparison.png", dpi=300)
        plt.close()

    # ==============================
    # BOXPLOTS
    # ==============================
    for metric in ["R2","RMSE","MSE","MAE"]:

        data_box = []
        labels = []

        for model in models:
            vals = [results[model][sr][metric]
                    for sr in scan_rates if results[model][sr] is not None]

            data_box.append(vals)
            labels.append(model)

        plt.figure(figsize=(14,7))

        box = plt.boxplot(data_box, labels=labels, patch_artist=True)

        colors = get_14_distinct_colors()
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        plt.xticks(rotation=30, ha='right', fontsize=9)
        plt.subplots_adjust(bottom=0.3)

        if metric == "R2":
            plt.title(r"$R^2$ Distribution")
        else:
            plt.title(f"{metric} Distribution")

        plt.grid(True)

        plt.savefig(f"{file_clean}/box_{metric}.png", dpi=300)
        plt.close()

    # ==============================
    # PER-MATERIAL METRICS TABLE (all models, all scan rates)
    # ==============================
    rows = []
    for model_name in models:
        for sr in scan_rates:
            res = results[model_name][sr]
            if res is not None:
                rows.append({
                    "Material": file_clean,
                    "Model": model_name,
                    "Scan Rate": sr,
                    "R2": res["R2"],
                    "RMSE": res["RMSE"],
                    "MAE": res["MAE"],
                    "MSE": res["MSE"]
                })
    pd.DataFrame(rows).to_excel(f"{file_clean}/all_model_metrics.xlsx", index=False)

# ==============================
# STACKING COMPARISON (SUBSCRIPT LEGEND)
# ==============================
def plot_stacking(metric, ylabel, filename):

    plt.figure(figsize=(10,6))
    colors = get_14_distinct_colors()

    for idx, file in enumerate(stacking_summary):
        vals = stacking_summary[file][metric]

        plt.plot(range(len(vals)), vals,
                 marker='o', linewidth=3,
                 color=colors[idx],
                 label=to_subscript(file))  # ✅ subscript

    if metric == "R2":
        plt.title(r"Stacking $R^2$ Comparison")
        plt.ylabel(r"$R^2$")
    else:
        plt.title(f"Stacking {metric} Comparison")
        plt.ylabel(ylabel)

    plt.xlabel("Scan Index")
    plt.grid(True)
    plt.legend()

    plt.savefig(filename, dpi=300)
    plt.show()

plot_stacking("R2", r"$R^2$", "Stacking_R2.png")
plot_stacking("RMSE", "RMSE", "Stacking_RMSE.png")
plot_stacking("MAE", "MAE", "Stacking_MAE.png")
plot_stacking("MSE", "MSE", "Stacking_MSE.png")

# ==============================
# DATASET SIZE TABLE (Supplementary Table requested by reviewer)
# ==============================
dataset_size_df = pd.DataFrame(dataset_size_records)
dataset_size_df.to_excel("dataset_sizes_per_material_scan_rate.xlsx", index=False)
print("\n📊 Dataset size table saved: dataset_sizes_per_material_scan_rate.xlsx")
print("🔧 Model hyperparameters/seed log saved: model_hyperparameters_and_seeds.json")

print("\n✅ FINAL SYSTEM COMPLETE (REPRODUCIBLE SEEDS + HARMONIZED HYPERPARAMETERS + REPORTING)")
