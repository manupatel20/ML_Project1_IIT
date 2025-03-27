# import os
# import csv
# import sys
# import numpy as np
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, r2_score

# # Ensure correct module import path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

# def test_predict():
#     model = LassoHomotopyModel()
#     data = []

#     # Use absolute path for the CSV file
#     test_file = os.path.join(os.path.dirname(__file__), "collinear_data.csv")  # Update filename as needed

#     # Read CSV data
#     with open(test_file, "r") as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             # Convert values to float and remove spaces from headers
#             data.append({key.strip(): float(value) for key, value in row.items() if value.strip()})

#     # Ensure data is not empty
#     assert len(data) > 0, "Error: CSV file is empty!"

#     # Print headers for debugging
#     # print("CSV Headers:", data[0].keys())

#     # Extract features (X) dynamically
#     feature_cols = [key for key in data[0].keys() if key.lower().startswith("x")]
#     assert len(feature_cols) > 0, "Error: No feature columns found in CSV!"

#     X = np.array([[datum[key] for key in feature_cols] for datum in data], dtype=float)

#     # Ensure X is not empty
#     assert X.shape[0] > 0 and X.shape[1] > 0, "Error: Feature matrix X is empty!"

#     # Determine target column dynamically
#     target_col = "y" if "y" in data[0] else "target" if "target" in data[0] else None
#     assert target_col is not None, "Error: Target column ('y' or 'target') is missing in CSV!"

#     y = np.array([datum[target_col] for datum in data], dtype=float)

#     '''    NEW CODE, UPDATE BELOW CODE ACCORDINGLY'''
#     # Split data into training (75%) and testing (25%) sets
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#     # Fit the model
#     results = model.fit(X_train, y_train)

#     # Make predictions
#     y_pred = results.predict(X_test)

#     # Fit the model
#     results = model.fit(X_train, y_train)
#     # print("Model fitting successful!")

#     zero_coeffs = np.sum(np.abs(results.coef_) < 1e-4)  # Count near-zero coefficients
#     total_coeffs = len(results.coef_)

#     print("Lambda:", results.mu)
#     print(f"Sparse Coefficients: {zero_coeffs}/{total_coeffs}")
#     print("Whole data trained Coefficients:")
#     print([float(f"{c:.1f}") if abs(c) >= 1e-9 else 0.0 for c in results.coef_])
#     mse = round(mean_squared_error(y_test, y_pred),4)
#     r2 = round(r2_score(y_test, y_pred),4)

#     print(f"Mean Squared Error: {mse}")
#     print(f"R² Score: {r2}")


#     # Load data again for verification
#     data1 = np.genfromtxt(test_file, delimiter=",", skip_header=1)
#     X1, y1 = data1[-2, :-1], data1[-2, -1]
#     X1 = np.array([X1])
#     y1 = np.array([y1])

#     X_prior = data1[:-2, :-1]
#     y_prior = data1[:-2, -1]
#     # print("X_prior:", X1)
#     # print("y_prior:", y1)

#     # Train the model
#     model1 = LassoHomotopyModel()
#     results1 = model1.fit(X_test, y_test)

#     # Make predictions
#     predictions = results1.predict(X1)
#     # print("Model Coefficients before sequencial learning:\n", model1.coef_)
#     # print("Predictions:", predictions, "\n")

#     # Verify sparsity of coefficients
#     # model2 = LassoHomotopyModel()
#     results2 = model1.update(X1, y1)
#     # print(results2)

#     zero_coeffs = np.sum(np.abs(results2.coef_) < 1e-4)  # Count near-zero coefficients
#     total_coeffs = len(results2.coef_)

#     print("\n")
#     print("Lambda:", results1.mu)
#     print(f"Sparse Coefficients: {zero_coeffs}/{total_coeffs}")
#     print("Model Coefficients after sequencial learning:")
#     print([float(f"{c:.1f}") if abs(c) >= 1e-9 else 0.0 for c in results1.coef_])

#     # Assertion for testing correctness (modify if needed)
#     assert isinstance(predictions, np.ndarray), "Error: Predictions are not in expected format!"
#     assert len(predictions) == len(y1), "Error: Number of predictions does not match y1!"

# if __name__ == "__main__":
#     test_predict()

'''
    # Split data into training (75%) and testing (25%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Fit the model
    results = model.fit(X_train, y_train)

    # Make predictions
    y_pred = results.predict(X_test)

    # Evaluate Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

    # Plot Lasso Path
    alphas, coefs = [], []
    for alpha in np.logspace(-3, 1, 50):
        model_temp = LassoHomotopyModel(mu=alpha)
        results_temp = model_temp.fit(X_train, y_train)
        alphas.append(alpha)
        coefs.append(results_temp.coef_)

    coefs = np.array(coefs)
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, coefs)
    plt.xscale("log")
    plt.xlabel("Alpha (Regularization Parameter)")
    plt.ylabel("Coefficients")
    plt.title("Lasso Homotopy Path")
    plt.show()'
    '''



import os
import csv
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Ensure correct module import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

def test_predict():
    data = []

    # Use absolute path for the CSV file
    test_file = os.path.join(os.path.dirname(__file__), "data_1.csv")  # Update filename as needed

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
    mse = round(mean_squared_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)

    print(f"Mean Squared Error: {mse}")
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

    mse = round(mean_squared_error(y_test, y_pred2), 4)
    r2 = round(r2_score(y_test, y_pred2), 4)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")