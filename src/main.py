import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

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
def get_models():
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200),
        "HistGB": HistGradientBoostingRegressor(),
        "SVR": SVR(),
        "KNN (5)": KNeighborsRegressor(5),
        "KNN (7)": KNeighborsRegressor(7),
        "KNN (9)": KNeighborsRegressor(9),
        "MLP": MLPRegressor(max_iter=2000),
        "Linear": LinearRegression(),
        "XGBoost": XGBRegressor(),
        "LightGBM": LGBMRegressor(),
        "CatBoost": CatBoostRegressor(verbose=0)
    }

    stacking = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(150)),
            ("gbr", GradientBoostingRegressor()),
            ("svr", SVR())
        ],
        final_estimator=LinearRegression()
    )

    models["Stacking"] = stacking
    return models

# ==============================
# STORAGE
# ==============================
stacking_summary = {}

# ==============================
# MAIN LOOP
# ==============================
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
            X, y, test_size=0.2, random_state=42
        )

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

            except:
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
            X, y, test_size=0.2, random_state=42
        )

        stack_model = StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(150)),
                ("gbr", GradientBoostingRegressor()),
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
    # BOXPLOTS (FIXED)
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

print("\n✅ FINAL SYSTEM COMPLETE (SUBSCRIPT + ALL FIXES)")
