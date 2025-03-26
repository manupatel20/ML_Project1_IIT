import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate approximately 1 MB of data
# Each row will have 9 features (x1 to x9) and 1 target (y)
# Assuming each row takes ~100 bytes, we need ~10,000 rows for 1 MB
num_rows = 100000

# Generate x1 and x3 as random numerical data
x1 = np.random.uniform(0, 100, size=num_rows)  # Random values between 0 and 100
x2 = np.random.uniform(0, 10, size=num_rows)  # Random values between 0 and 10
x3 = np.random.uniform(0, 100, size=num_rows)  # Random values between 0 and 100
x4 = np.random.uniform(0, 100, size=num_rows)  # Random values between 0 and 100
x5 = np.random.uniform(0, 10, size=num_rows)  # Random values between 0 and 10
x6 = np.random.uniform(0, 100, size=num_rows)  # Random values between 0 and 100
x7 = np.random.uniform(0, 1000, size=num_rows)  # Random values between 0 and 1000
x8 = np.random.uniform(0, 1000, size=num_rows)  # Random values between 0 and 1000
x9 = np.random.uniform(0, 100, size=num_rows)  # Random values between 0 and 100



# Compute the target y = (x1 + x3) * x9
y = (x1 + x2 - x3 * x4) * x5 - x6 + - x7 * x8 +x9

# Create a DataFrame to store the data
data = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'x3': x3,
    'x4': x4,
    'x5': x5,
    'x6': x6,
    'x7': x7,
    'x8': x8,
    'x9': x9,
    'y': y
})

# Save the DataFrame to a CSV file
data.to_csv('data_1.csv', index=False)

print("Dataset generated and saved as 'data_4.csv'.")
print(f"Dataset size: {data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")