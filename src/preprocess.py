"""
Data Preprocessing & Feature Engineering Module
Project: Used Car Resale Price Prediction Using Machine Learning Algorithms

Purpose:
--------
1. Load the raw CarDekho dataset ('data/Car details v3.csv').
2. Remove duplicate rows to prevent data leakage.
3. Clean unit-based text columns ('mileage', 'engine', 'max_power') to numeric floats.
4. Extract the 'brand' feature from the full car name.
5. Impute missing values and replace physically invalid zeros (e.g. 0 kmpl mileage).
6. Build a reusable Scikit-Learn ColumnTransformer for One-Hot Encoding and Scaling.
7. Split the cleaned data into 80% Training and 20% Testing sets.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Supported top car brands (brands with >= 10 cars in dataset); rare ones grouped into 'Other'
TOP_BRANDS = [
    'Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Honda', 'Ford',
    'Toyota', 'Chevrolet', 'Renault', 'Volkswagen', 'Nissan',
    'Skoda', 'Datsun', 'BMW', 'Mercedes-Benz', 'Fiat', 'Audi',
    'Jeep', 'Mitsubishi'
]


def extract_numeric(value):
    """
    Extracts the first numeric float from a string containing text and units.
    Examples:
        '23.4 kmpl'  -> 23.4
        '1248 CC'    -> 1248.0
        '74 bhp'     -> 74.0
        ' bhp'       -> np.nan
        np.nan       -> np.nan
    """
    if pd.isna(value):
        return np.nan
    
    # Use regular expression to find the first floating-point or integer number
    match = re.search(r'[-+]?\d*\.\d+|\d+', str(value))
    if match:
        return float(match.group())
    return np.nan


def clean_brand(name):
    """
    Extracts the car make/brand from the full vehicle name.
    If the brand has very few occurrences in the dataset, categorizes it as 'Other'.
    """
    if pd.isna(name):
        return 'Other'
    brand = str(name).strip().split()[0]
    return brand if brand in TOP_BRANDS else 'Other'


def load_and_clean_data(file_path):
    """
    Loads raw CSV, cleans text units, handles missing values, and drops duplicates.
    
    Returns:
        pd.DataFrame: Cleaned and structured dataframe.
    """
    print(f"[Preprocessing] Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    initial_rows = len(df)
    
    # Step 1: Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[Preprocessing] Dropped {initial_rows - len(df)} duplicate rows. Remaining: {len(df)}")
    
    # Step 2: Feature Engineering - Extract Brand
    df['brand'] = df['name'].apply(clean_brand)
    
    # Step 3: Clean columns with text units into numeric floats
    df['mileage'] = df['mileage'].apply(extract_numeric)
    df['engine'] = df['engine'].apply(extract_numeric)
    df['max_power'] = df['max_power'].apply(extract_numeric)
    
    # Step 4: Drop 'torque' (highly irregular formats, captured by engine & power)
    # and drop raw 'name' (already captured via 'brand')
    df = df.drop(columns=['name', 'torque'], errors='ignore')
    
    # Step 5: Handle missing values in critical technical features
    # Rows missing technical specs are dropped to preserve authentic data quality
    before_drop_na = len(df)
    df = df.dropna(subset=['mileage', 'engine', 'max_power', 'seats']).reset_index(drop=True)
    print(f"[Preprocessing] Dropped {before_drop_na - len(df)} rows missing technical specs. Remaining: {len(df)}")
    
    # Step 6: Handle physical invalidities (e.g., mileage == 0 kmpl)
    median_mileage = df.loc[df['mileage'] > 0, 'mileage'].median()
    df['mileage'] = df['mileage'].replace(0, median_mileage)
    
    # Convert 'seats' to integer
    df['seats'] = df['seats'].astype(int)
    
    return df


def get_feature_preprocessor(categorical_features, numeric_features):
    """
    Creates a Scikit-Learn ColumnTransformer that:
    - Scales numeric columns with StandardScaler (mean=0, variance=1)
    - Encodes categorical columns with OneHotEncoder
    
    Using ColumnTransformer ensures preprocessing is performed consistently
    without data leakage between train and test sets.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


def prepare_datasets(raw_csv_path="data/Car details v3.csv", test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline:
    - Cleans raw data
    - Saves cleaned dataset to 'data/cleaned_car_data.csv'
    - Splits into Train and Test sets
    
    Returns:
        X_train, X_test, y_train, y_test, preprocessor, feature_info
    """
    cleaned_df = load_and_clean_data(raw_csv_path)
    
    # Save clean dataset for college record / inspection
    cleaned_csv_path = os.path.join("data", "cleaned_car_data.csv")
    cleaned_df.to_csv(cleaned_csv_path, index=False)
    print(f"[Preprocessing] Cleaned dataset saved to: {cleaned_csv_path}")
    
    # Separate Features (X) and Target (y)
    X = cleaned_df.drop(columns=['selling_price'])
    y = cleaned_df['selling_price']
    
    # Define column types
    categorical_features = ['brand', 'fuel', 'seller_type', 'transmission', 'owner']
    numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    
    # Build preprocessor
    preprocessor = get_feature_preprocessor(categorical_features, numeric_features)
    
    # 80-20 Train-Test Split with reproducible random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    feature_info = {
        'categorical_features': categorical_features,
        'numeric_features': numeric_features,
        'brands': sorted(cleaned_df['brand'].unique().tolist()),
        'fuel_types': sorted(cleaned_df['fuel'].unique().tolist()),
        'seller_types': sorted(cleaned_df['seller_type'].unique().tolist()),
        'transmission_types': sorted(cleaned_df['transmission'].unique().tolist()),
        'owner_types': sorted(cleaned_df['owner'].unique().tolist()),
        'year_min': int(cleaned_df['year'].min()),
        'year_max': int(cleaned_df['year'].max()),
        'seats_options': sorted(cleaned_df['seats'].unique().tolist())
    }
    
    print(f"[Preprocessing] Train set size: {X_train.shape[0]} samples")
    print(f"[Preprocessing] Test set size:  {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, preprocessor, feature_info


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor, feature_info = prepare_datasets()
    print("\nFeature Information Extracted:")
    for key, val in feature_info.items():
        print(f"  {key}: {val}")
