import os
import csv
import sys
import numpy as np

# Ensure correct module import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

def test_predict():
    model = LassoHomotopyModel()
    data = []

    # Use absolute path for the CSV file
    test_file = os.path.join(os.path.dirname(__file__), "non-correlated_data.csv")  # Update filename as needed

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

    # Fit the model
    results = model.fit(X, y)
    # print("Model fitting successful!")

    zero_coeffs = np.sum(np.abs(results.coef_) < 1e-4)  # Count near-zero coefficients
    total_coeffs = len(results.coef_)

    print("Lambda:", results.mu)
    print(f"Sparse Coefficients: {zero_coeffs}/{total_coeffs}")
    print("Whole data trained Coefficients:")
    print([float(f"{c:.1f}") if abs(c) >= 1e-9 else 0.0 for c in results.coef_])

    # Load data again for verification
    data1 = np.genfromtxt(test_file, delimiter=",", skip_header=1)
    X1, y1 = data1[-2, :-1], data1[-2, -1]
    X1 = np.array([X1])
    y1 = np.array([y1])

    X_prior = data1[:-2, :-1]
    y_prior = data1[:-2, -1]
    # print("X_prior:", X1)
    # print("y_prior:", y1)

    # Train the model
    model1 = LassoHomotopyModel()
    results1 = model1.fit(X_prior, y_prior)

    # Make predictions
    predictions = results1.predict(X1)
    # print("Model Coefficients before sequencial learning:\n", model1.coef_)
    # print("Predictions:", predictions, "\n")

    # Verify sparsity of coefficients
    # model2 = LassoHomotopyModel()
    results2 = model1.update(X1, y1)
    # print(results2)

    zero_coeffs = np.sum(np.abs(results2.coef_) < 1e-4)  # Count near-zero coefficients
    total_coeffs = len(results2.coef_)

    print("\n")
    print("Lambda:", results1.mu)
    print(f"Sparse Coefficients: {zero_coeffs}/{total_coeffs}")
    print("Model Coefficients after sequencial learning:")
    print([float(f"{c:.1f}") if abs(c) >= 1e-9 else 0.0 for c in results1.coef_])

    # Assertion for testing correctness (modify if needed)
    assert isinstance(predictions, np.ndarray), "Error: Predictions are not in expected format!"
    assert len(predictions) == len(y1), "Error: Number of predictions does not match y1!"

if __name__ == "__main__":
    test_predict()