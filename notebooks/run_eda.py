"""
Exploratory Data Analysis (EDA) Script
Project: Used Car Resale Price Prediction Using Machine Learning Algorithms
Purpose: Inspect dataset, compute statistical summaries, and generate visual plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'Arial'

DATA_PATH = os.path.join("data", "Car details v3.csv")
OUTPUT_DIR = os.path.join("notebooks", "eda_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("1. LOADING DATASET")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

print("=" * 60)
print("2. BASIC INFORMATION & DATA TYPES")
print("=" * 60)
print(df.info())

print("\n" + "=" * 60)
print("3. MISSING VALUES PER COLUMN")
print("=" * 60)
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n" + "=" * 60)
print("4. DUPLICATE ROWS")
print("=" * 60)
dup_count = df.duplicated().sum()
print(f"Total duplicate rows: {dup_count} ({(dup_count / len(df)) * 100:.2f}%)")

print("\n" + "=" * 60)
print("5. NUMERICAL SUMMARY STATISTICS")
print("=" * 60)
print(df.describe().T)

print("\n" + "=" * 60)
print("6. GENERATING EDA PLOTS")
print("=" * 60)

# Helper function to extract numeric values for EDA correlation
def extract_numeric(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    parts = val_str.split()
    if not parts:
        return np.nan
    try:
        return float(parts[0])
    except ValueError:
        return np.nan

# Quick temporary numeric extraction strictly for EDA visual correlation
eda_df = df.copy()
eda_df['mileage_num'] = eda_df['mileage'].apply(extract_numeric)
eda_df['engine_num'] = eda_df['engine'].apply(extract_numeric)
eda_df['max_power_num'] = eda_df['max_power'].apply(extract_numeric)
eda_df['brand'] = eda_df['name'].apply(lambda x: str(x).split()[0])

# Plot 1: Target Variable Distribution (Selling Price)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(eda_df['selling_price'] / 1e5, kde=True, ax=ax[0], color='royalblue', bins=35)
ax[0].set_title("Selling Price Distribution (Lakh ₹)", fontsize=13, fontweight='bold')
ax[0].set_xlabel("Selling Price (Lakh ₹)")
ax[0].set_ylabel("Count")

sns.boxplot(x=eda_df['selling_price'] / 1e5, ax=ax[1], color='lightsteelblue')
ax[1].set_title("Selling Price Boxplot (Outlier Detection)", fontsize=13, fontweight='bold')
ax[1].set_xlabel("Selling Price (Lakh ₹)")

plt.tight_layout()
p1 = os.path.join(OUTPUT_DIR, "01_selling_price_distribution.png")
fig.savefig(p1, dpi=300)
plt.close(fig)
print(f"Saved: {p1}")

# Plot 2: Price by Fuel Type and Transmission
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=eda_df, x='fuel', y=eda_df['selling_price'] / 1e5, ax=ax[0], palette="Set2")
ax[0].set_title("Resale Price vs Fuel Type", fontsize=13, fontweight='bold')
ax[0].set_xlabel("Fuel Type")
ax[0].set_ylabel("Selling Price (Lakh ₹)")
ax[0].set_ylim(0, 35)  # Zoom to focus on majority of data points

sns.boxplot(data=eda_df, x='transmission', y=eda_df['selling_price'] / 1e5, ax=ax[1], palette="Pastel1")
ax[1].set_title("Resale Price vs Transmission", fontsize=13, fontweight='bold')
ax[1].set_xlabel("Transmission")
ax[1].set_ylabel("Selling Price (Lakh ₹)")
ax[1].set_ylim(0, 45)

plt.tight_layout()
p2 = os.path.join(OUTPUT_DIR, "02_price_by_fuel_and_transmission.png")
fig.savefig(p2, dpi=300)
plt.close(fig)
print(f"Saved: {p2}")

# Plot 3: Correlation Heatmap for Numerical Features
num_cols = ['year', 'selling_price', 'km_driven', 'mileage_num', 'engine_num', 'max_power_num', 'seats']
corr = eda_df[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, ax=ax)
ax.set_title("Correlation Heatmap (Numerical Features)", fontsize=14, fontweight='bold')
plt.tight_layout()
p3 = os.path.join(OUTPUT_DIR, "03_correlation_heatmap.png")
fig.savefig(p3, dpi=300)
plt.close(fig)
print(f"Saved: {p3}")

# Plot 4: Top 10 Car Brands by Frequency
fig, ax = plt.subplots(figsize=(10, 5))
top_brands = eda_df['brand'].value_counts().head(10)
sns.barplot(x=top_brands.values, y=top_brands.index, palette="viridis", ax=ax)
ax.set_title("Top 10 Most Common Car Brands in Dataset", fontsize=13, fontweight='bold')
ax.set_xlabel("Number of Cars")
ax.set_ylabel("Brand")
for i, v in enumerate(top_brands.values):
    ax.text(v + 20, i, str(v), color='black', va='center', fontweight='bold')

plt.tight_layout()
p4 = os.path.join(OUTPUT_DIR, "04_top_car_brands.png")
fig.savefig(p4, dpi=300)
plt.close(fig)
print(f"Saved: {p4}")

print("\nEDA Completed successfully!")
