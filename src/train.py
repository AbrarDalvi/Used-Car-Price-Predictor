"""
Model Training, Evaluation & Comparison Module
Project: Used Car Resale Price Prediction Using Machine Learning Algorithms

Purpose:
--------
1. Load and split preprocessed data via `src.preprocess.prepare_datasets`.
2. Construct Scikit-Learn Pipelines for three ensemble regressors:
   - Random Forest Regressor
   - Extra Trees Regressor
   - Gradient Boosting Regressor
3. Train all three models on the exact same training set (5,373 samples).
4. Evaluate all three models on the unseen test set (1,344 samples) using:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Coefficient of Determination (R²)
5. Log genuine, non-fabricated metrics.
6. Save comparison plots and serialize the best-performing pipeline for deployment.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path to allow execution from any directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from src.preprocess import prepare_datasets
except ImportError:
    from preprocess import prepare_datasets

# Directories
MODELS_DIR = "models"
PLOTS_DIR = os.path.join("notebooks", "eda_plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def build_pipeline(preprocessor, regressor):
    """
    Combines the feature preprocessor and the regression algorithm into
    a single unified Scikit-Learn Pipeline.
    """
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])


def train_and_evaluate_models():
    """
    Trains and compares Random Forest, Extra Trees, and Gradient Boosting regressors.
    """
    print("=" * 70)
    print("USED CAR RESALE PRICE PREDICTION - MODEL BENCHMARKING")
    print("=" * 70)
    
    # 1. Prepare Data
    X_train, X_test, y_train, y_test, preprocessor, feature_info = prepare_datasets()
    
    # 2. Define the Three Required Machine Learning Algorithms
    models_to_train = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        ),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }
    
    results = []
    trained_pipelines = {}
    predictions_dict = {}
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING & EVALUATION (Test Set: 1,344 samples)")
    print("=" * 70)
    
    for name, regressor in models_to_train.items():
        print(f"\n[Training] {name}...")
        pipeline = build_pipeline(preprocessor, regressor)
        
        # Measure training duration
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predictions on unseen test set
        y_pred = pipeline.predict(X_test)
        
        # Calculate genuine evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results.append({
            "Model": name,
            "MAE (INR)": round(mae, 2),
            "RMSE (INR)": round(rmse, 2),
            "R2 Score": round(r2, 4),
            "Training Time (s)": round(training_time, 2)
        })
        
        trained_pipelines[name] = pipeline
        predictions_dict[name] = y_pred
        
        print(f"  ✓ MAE:  ₹{mae:,.2f}")
        print(f"  ✓ RMSE: ₹{rmse:,.2f}")
        print(f"  ✓ R²:   {r2:.4f} ({r2 * 100:.2f}%)")
        print(f"  ✓ Time: {training_time:.2f} seconds")
    
    # 3. Create Comparison DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # 4. Identify Best Performing Model
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    best_r2 = results_df.iloc[0]["R2 Score"]
    print(f"\n🏆 Best Model Selected: {best_model_name} (R² = {best_r2})")
    
    # 5. Save the Best Model Pipeline
    model_save_path = os.path.join(MODELS_DIR, "best_car_price_model.joblib")
    joblib.dump(best_pipeline, model_save_path)
    print(f"[Export] Saved best model pipeline to: {model_save_path}")
    
    # Save comparison table as JSON
    results_json_path = os.path.join(MODELS_DIR, "model_comparison.json")
    results_df.to_json(results_json_path, orient="records", indent=2)
    print(f"[Export] Saved comparison table to: {results_json_path}")
    
    # Save metadata for Streamlit Web UI
    metadata = {
        "best_model_name": best_model_name,
        "best_model_r2": float(best_r2),
        "metrics_summary": results_df.to_dict(orient="records"),
        "feature_info": feature_info
    }
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Export] Saved application metadata to: {metadata_path}")
    
    # 6. Generate Visual Performance Comparison Plots
    generate_evaluation_plots(results_df, y_test, predictions_dict, best_model_name)
    
    return results_df, best_model_name


def generate_evaluation_plots(results_df, y_test, predictions_dict, best_model_name):
    """
    Generates comparison bar charts and an Actual vs Predicted scatter plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot A: Metric Comparison Bar Chart (R² Score)
    sns.barplot(
        data=results_df,
        x="Model",
        y="R2 Score",
        hue="Model",
        legend=False,
        palette="viridis",
        ax=axes[0]
    )
    axes[0].set_title("Model Comparison: R² Score (Higher is Better)", fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("R² Score")
    axes[0].set_xlabel("")
    for i, row in results_df.iterrows():
        axes[0].text(i, row["R2 Score"] + 0.02, f"{row['R2 Score']:.4f}", ha='center', fontweight='bold')
        
    # Plot B: Actual vs Predicted Selling Prices for Best Model
    y_test_lakh = y_test / 1e5
    best_preds_lakh = predictions_dict[best_model_name] / 1e5
    
    sns.scatterplot(
        x=y_test_lakh,
        y=best_preds_lakh,
        alpha=0.5,
        color="teal",
        ax=axes[1]
    )
    # Ideal 45-degree reference line
    max_val = min(y_test_lakh.max(), 50)
    axes[1].plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label="Ideal Fit (y = x)")
    axes[1].set_xlim(0, max_val)
    axes[1].set_ylim(0, max_val)
    axes[1].set_title(f"Actual vs Predicted Resale Price ({best_model_name})", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Actual Price (Lakh ₹)")
    axes[1].set_ylabel("Predicted Price (Lakh ₹)")
    axes[1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "05_model_performance_comparison.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved model evaluation visual to: {plot_path}")


if __name__ == "__main__":
    train_and_evaluate_models()
