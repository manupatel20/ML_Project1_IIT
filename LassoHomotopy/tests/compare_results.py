
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLars
from sklearn.linear_model import Lasso  # Use Lasso instead of LassoLars

import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.LassoHomotopy import LassoHomotopyModel

import csv

# Load dataset
data = []
with open("test_1.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

# X = np.array([[v for k,v in datum.items() if k.startswith('x')] for datum in data], dtype=float)
feature_cols = [key for key in data[0].keys() if key.lower().startswith("x")]
assert len(feature_cols) > 0, "Error: No feature columns found in CSV!"
X = np.array([[datum[key] for key in feature_cols] for datum in data], dtype=float)

    # Ensure X is not empty
assert X.shape[0] > 0 and X.shape[1] > 0, "Error: Feature matrix X is empty!"

    # Determine target column dynamically
target_col = "y" if "y" in data[0] else "target" if "target" in data[0] else None
assert target_col is not None, "Error: Target column ('y' or 'target') is missing in CSV!"

y = np.array([datum[target_col] for datum in data], dtype=float)
# y = np.array([[v for k,v in datum.items() if k=='y'] for datum in data], dtype=float)

# Initialize models
homotopy_model = LassoHomotopyModel()
lars_model = LassoLars(alpha=0.1)  # alpha = lambda / n_samples
lasso_model = Lasso(alpha=0.1)


# Fit models
homotopy_model.fit(X, y)
lars_model.fit(X, y)
lasso_model.fit(X, y)

# Compare coefficients
print("HomotopyLASSO Coefficients:\n", homotopy_model.coef_, "\n\n")
print("LassoLars Coefficients:\n", lars_model.coef_, "\n\n")
print("Scikit-learn Lasso Coefficients:\n", lasso_model.coef_, "\n\n")