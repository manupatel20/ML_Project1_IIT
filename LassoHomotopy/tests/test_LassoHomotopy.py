import os
import csv
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Ensure correct module import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

def test_predict():
    data = []

    # Use absolute path for the CSV file
    test_file = os.path.join(os.path.dirname(__file__), "collinear_data.csv")  # Update filename as needed

    # Read CSV data
    with open(test_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert values to float and remove spaces from headers
            data.append({key.strip(): float(value) for key, value in row.items() if value.strip()})

    # Ensure data is not empty
    assert len(data) > 0, "Error: CSV file is empty!"

    # Print headers for debugging
    # print("CSV Headers:", data[0].keys())

    # Extract features (X) dynamically
    feature_cols = [key for key in data[0].keys() if key.lower().startswith("x")]
    assert len(feature_cols) > 0, "Error: No feature columns found in CSV!"

    X = np.array([[datum[key] for key in feature_cols] for datum in data], dtype=float)

    # Ensure X is not empty
    assert X.shape[0] > 0 and X.shape[1] > 0, "Error: Feature matrix X is empty!"

    # Determine target column dynamically
    target_col = "y" if "y" in data[0] else "target" if "target" in data[0] else None
    assert target_col is not None, "Error: Target column ('y' or 'target') is missing in CSV!"

    y = np.array([datum[target_col] for datum in data], dtype=float)

    #### MODEL 1 ####
    # Define split index (75% for training, 25% for testing)
    split_index = int(0.75 * len(X))  # 75% of data

    # Manually split data for 1st model
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    
    model1 = LassoHomotopyModel()

    # Fit the model
    results = model1.fit(X_train, y_train)

    # Make predictions
    y_pred = results.predict(X_test)

    zero_coeffs = np.sum(np.abs(results.coef_) < 1e-4)  # Count near-zero coefficients
    total_coeffs = len(results.coef_)

    print("\n\nModel 1")
    print("Lambda:", results.mu)
    print(f"Sparse Coefficients: {zero_coeffs}/{total_coeffs}")
    print("Whole data trained Coefficients:")
    print([float(f"{c:.1f}") if abs(c) >= 1e-9 else 0.0 for c in results.coef_])
    
    r2 = round(r2_score(y_test, y_pred), 4)
    print(f"R² Score: {r2}")












    #### MODEL 2 ####
    # Define split indices for 2nd model
    train_index = int(0.75 * len(X))  # First 75% for training
    valid_index = int(0.90 * len(X))  # Next 15% for validation, last 10% for testing

    # Manual Splitting
    X_train, y_train = X[:train_index], y[:train_index]      # First 75%
    X_val, y_val = X[train_index:valid_index], y[train_index:valid_index]  # Next 15%
    X_test, y_test = X[valid_index:], y[valid_index:]        # Last 10%

    # Fit the model
    model2 = LassoHomotopyModel()
    results21 = model2.fit(X_train, y_train)

    for i in range(X_val.shape[0]):
        results22 = model2.update(X_val[i], y_val[i])
    # results22 = model2.update(X_val, y_val)

    # Make predictions
    y_pred2 = results22.predict(X_test)

    zero_coeffs = np.sum(np.abs(results22.coef_) < 1e-4)      # Count near-zero coefficients
    total_coeffs = len(results22.coef_)
    print("\nModel 2 - Sequential Learning")
    print("Lambda:", results22.mu)
    print(f"Sparse Coefficients: {zero_coeffs}/{total_coeffs}")
    print("Model Coefficients after sequencial learning:")      # Coefficients after sequential learning
    print([float(f"{c:.1f}") if abs(c) >= 1e-9 else 0.0 for c in results22.coef_])    # Coefficients after sequential learning

    r2 = round(r2_score(y_test, y_pred2), 4)
    print(f"R² Score: {r2}")