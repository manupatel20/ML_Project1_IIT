# Project 1 ( Given Problem For Project )

Your objective is to implement the LASSO regularized regression model using the Homotopy Method. You can read about this method in [this](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf) paper and the references therein. You are required to write a README for your project. Please describe how to run the code in your project *in your README*. Including some usage examples would be an excellent idea. You may use Numpy/Scipy, but you may not use built-in models from, e.g. SciKit Learn. This implementation must be done from first principles. You may use SciKit Learn as a source of test data.

You should create a virtual environment and install the packages in the requirements.txt in your virtual environment. You can read more about virtual environments [here](https://docs.python.org/3/library/venv.html). Once you've installed PyTest, you can run the `pytest` CLI command *from the tests* directory. I would encourage you to add your own tests as you go to ensure that your model is working as a LASSO model should (Hint: What should happen when you feed it highly collinear data?)

In order to turn your project in: Create a fork of this repository, fill out the relevant model classes with the correct logic. Please write a number of tests that ensure that your LASSO model is working correctly. It should produce a sparse solution in cases where there is collinear training data. You may check small test sets into GitHub, but if you have a larger one (over, say 20MB), please let us know and we will find an alternative solution. In order for us to consider your project, you *must* open a pull request on this repo. This is how we consider your project is "turned in" and thus we will use the datetime of your pull request as your submission time. If you fail to do this, we will grade your project as if it is late, and your grade will reflect this as detailed on the course syllabus. 

You may include Jupyter notebooks as visualizations or to help explain what your model does, but you will be graded on whether your model is correctly implemented in the model class files and whether we feel your test coverage is adequate. We may award bonus points for compelling improvements/explanations above and beyond the assignment.


---

## Table of Contents

- [Questions](#questions)
- [Installation and Setup](#installation-and-setup)
- [Testing Different Datasets](#testing-different-datasets)
- [Compare.py and generate.py](#comparepy-and-generatepy)
- [LASSO Regression using the Homotopy Algorithm ( Extra Details )](#lasso-regression-using-the-homotopy-algorithm--extra-details-)
  - [Introduction](#introduction)
  - [What is LASSO Regression?](#what-is-lasso-regression)
  - [Comparison Between Linear Regression and LASSO](#comparison-between-linear-regression-and-lasso)
  - [What is the Homotopy Algorithm?](#what-is-the-homotopy-algorithm)
  - [Conclusion](#conclusion)
  - [References](#references)

---
---

# Questions

Put your README here. Answer the following questions.

* What does the model you have implemented do and when should it be used?

    The model implements **LASSO regression using the Homotopy Algorithm**. It is particularly useful for:
  - **Feature Selection**: Eliminates irrelevant features by setting some coefficients to zero.
  - **Handling Multicollinearity**: Works well in situations where predictors are highly correlated.
  - **Sparse Models**: Reduces model complexity, making it interpretable and efficient.
  - **High-Dimensional Data**: When the number of features is much larger than the number of observations.


* How did you test your model to determine if it is working reasonably correctly?
  - **Unit Tests**: Using `pytest`, we wrote tests to validate:
  - Correct computation of cost function.
  - Proper coefficient shrinkage for different values of \( \lambda \).
  - The Homotopy path follows expected trajectories.
  - **Synthetic Data**: Generated datasets where the expected outcome was known.
  - **Comparison with Scikit-Learn**: Compared the results against `sklearn.linear_model.Lasso` to verify correctness.

* What parameters have you exposed to users of your implementation in order to tune performance?
  - `lambda (\lambda)`: Controls the degree of regularization.
  - `tolerance`: Convergence threshold for optimization.
  - `max_iter`: Limits the number of iterations for efficiency.
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
   - **Handling Extremely Large Datasets**: Current implementation may struggle with very large feature spaces. **Possible fix:** Use batch processing or sparse matrix optimizations.
   - **Highly Collinear Features**: While LASSO selects one feature among correlated ones, it may be unstable. **Possible fix:** Combine LASSO with Ridge regression (**Elastic Net**).
   - **Path Efficiency**: Computing the full Homotopy path may be expensive. **Possible fix:** Use an adaptive approach to select relevant \( \lambda \) values dynamically.




------
---

## Installation and Setup

### 1. Clone Repository
To ensure dependency management, create a virtual environment:
```sh
git clone https://github.com/manupatel20/ML_Project1_IIT
```

### 2. Create a Virtual Environment
To ensure dependency management, create a virtual environment:
for Windows
```sh
cd .\ML_Project1_IIT\LassoHomotopy\
python -m venv .env
.env\scripts\activate
```
for Mac
```sh
cd .\ML_Project1_IIT\LassoHomotopy\
python -m venv .env
source .env/bin/activate
```

### 3. Install Dependencies
Ensure that you have all required dependencies installed by running:
```sh
cd ..
pip install -r requirements.txt
```

---

### 4. Running the Project
Run the following command to train and test the LASSO model using the Homotopy method:
```sh
cd .\LassoHomotopy\tests\
pytest .\test_LassoHomotopy.py
```
![image](https://github.com/user-attachments/assets/5fa974e4-da0e-41d7-b425-8ed2a1799a5c)

where:
- `small_test.csv` is the input dataset.
- test from different files by changing file names in `test_LassoHomotopy.py`

### 5. Evaluating the Model
To evaluate the trained LASSO model, run:
```sh
python .\test_LassoHomotopy.py
```
![image](https://github.com/user-attachments/assets/b335696b-fad7-4de3-a154-7b55e63bd11c)

---
---

## Testing Different Datasets

We have tested different datasets including dataset with high colinearity and low colinearity. Let us observe the details of data and result.


| **Data File**  | **Details** |  **Colinear** | **Sparse Solution** | **Sparse Var**  | **Output** |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| non-correlated_data.csv  | All correlations between x1, x2, x3, and x4 are very weak  | No  | No | 0/4  | ![image](https://github.com/user-attachments/assets/0428a2e5-71b6-451d-b66a-a77c1c6abb59) |
| test_1.csv  |X_5, X_2, X_15, and X_3 are the most influential features for the target. X_1 and X_7 negatively affect it, while others (e.g., X_6, X_10) have minimal impact. X_5 and X_15: Moderate positive correlation (~0.50). As X_5 increases, X_15 tends to increase. X_6 and X_10: Strong positive correlation (~0.80). These two features are highly linearly related. X_2 and X_5: Moderate positive correlation (~0.60). Most other pairs (e.g., X_1 vs. X_3, X_7 vs. X_10) show weak correlations < 0.4 | Slightly(7-10)  | Moderate | 8/18  | ![image](https://github.com/user-attachments/assets/4709f781-c8fa-4a01-a1da-2bc1c29fe275) |
| test_2.csv  | x5, x3, x4, x6, and x7 are the primary drivers of y, all negatively. x2, x8, and x1 have lesser influence; x9 is nearly irrelevant linearly. x3 and x4: Perfect correlation (1.00). These are essentially identical (x4 = x3 / 2). x6 and x7: Perfect correlation (1.00). These are identical (x7 = x6 / 2). x3, x4, and x5: Very strong correlation (~0.95). x5 closely follows x3 and x4. x3, x4, x5, x6, x7: Moderate to strong correlations (~0.60-0.65), suggesting a cluster of related features. Other pairs (e.g., x1-x2, x8-x9) show weak correlations < 0.25  | Highly  | Yes | 5/9  | ![image](https://github.com/user-attachments/assets/5d96cd87-b07c-42ff-a115-e55cf7b170b8) |
| data_1.csv  | x4, x3, and x6 are the primary drivers of y. All pairwise correlations between features are weak 0.10, indicating no significant collinearity. Features x1 to x9 appear largely independent of each other. | Highly  | Yes | 7/9  | ![image](https://github.com/user-attachments/assets/30df637b-3aa1-4bbe-b6d2-6cd319277a01) |
| data_2.csv  | x5-x9 dominate y’s variation positively, x1/x2 oppose it strongly, and x3/x4 have a moderate positive effect.x1 and x2: Perfect correlation (1.00). Likely x2 = x1/2. x3 and x4: Perfect correlation (1.00). Likely x4 = x3/2. x5 to x9: Perfect correlation (1.00). These are identical variables. x1/x2 vs. x5-x9: Strong negative correlation (-0.90), indicating an inverse relationship. x3/x4 vs. x5-x9: Moderate positive correlation (0.51). x1/x2 vs. x3/x4: Weak correlation (0.06), suggesting independence. | Moderately  | Moderate | 4/9  | ![image](https://github.com/user-attachments/assets/61d49c64-5483-4e4d-9e59-4d2de19a840e) |
---
---

## compare.py and generate.py


---
---

# LASSO Regression using the Homotopy Algorithm ( Extra Details )

## Introduction
This project implements the **LASSO regression model** using the **Homotopy Method** from first principles using **NumPy**. LASSO is a powerful technique for regression analysis that introduces an L1 regularization term to promote sparsity in the model parameters. The Homotopy algorithm efficiently finds solutions to the LASSO problem as the regularization parameter varies.


---

## What is LASSO Regression?
LASSO regression is a type of linear regression that includes an L1 penalty on the absolute values of the regression coefficients. The main advantages of LASSO include:

1. **Feature Selection**: Some regression coefficients shrink to exactly zero, eliminating irrelevant features.
2. **Sparsity**: It creates a sparse model that is easier to interpret.
3. **Prevention of Overfitting**: Regularization controls model complexity and improves generalization.

LASSO regression minimizes the following objective function:

![image](https://github.com/user-attachments/assets/666ac99c-a31a-4668-a723-f56d62956fd8)


### Comparison Between Linear Regression and LASSO
The following images illustrate the difference between **ordinary linear regression** and **LASSO regression with regularization**:

**Linear Regression:**

![linear reg](https://github.com/user-attachments/assets/b6eae11d-1404-4393-b8f9-8781c850ef3e)

Here Lambda=0 means it is linear regression.

**Ridge Regression: L2 vs Lasso L1**

![image](https://github.com/user-attachments/assets/0ff5a95d-88ae-481d-b312-39c243878fb3)

Lambda* slope^2 (left side) is how Ridge works for different values of lambda and

Lambda* |slope| (right side) is how Lasso works for different values of lambda


**LASSO Regularization Effect:**

![4](https://github.com/user-attachments/assets/7b1c4615-5c98-4fb1-93b7-6c5f1ae36740)


From the images, we see that increasing the **regularization parameter (\lambda)** forces the slope to shrink toward zero, preventing large coefficient values.

---

## What is the Homotopy Algorithm?
The **Homotopy Algorithm** is an efficient method for solving the LASSO problem by tracking the solution path as the regularization parameter ( \lambda \) changes. Instead of solving the LASSO problem independently for multiple values of ( \lambda \), the Homotopy algorithm exploits the structure of the optimization problem to compute an entire path of solutions in an efficient manner.

**In simple terms we update regularization parameter and then compute for t=0 to t=1 using updated param.**
![name](https://github.com/user-attachments/assets/28038d83-85a4-4b8c-9017-5d03fc35dc31)

**RecLasso: homotopy algorithm for Lasso**

![image](https://github.com/user-attachments/assets/7ec4503c-7e9c-43b0-97bf-7219fd56aa1f)



Key advantages of the Homotopy Algorithm:
- **Computational Efficiency**: Faster than solving multiple LASSO problems independently.
- **Path Tracking**: Provides a complete solution trajectory for different values of \( \lambda \).
- **Optimal for Sparse Solutions**: Particularly useful when the expected solution is sparse.

For further theoretical background, refer to the uploaded research paper:[Link](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf)

---
---

## Conclusion
This project successfully implements LASSO regression using the Homotopy Algorithm. The model efficiently tracks the solution path as the regularization parameter varies, making it a powerful tool for feature selection and sparse modeling. By testing on synthetic and real datasets, we ensure correctness and demonstrate practical applications of LASSO regression.


---
---

## References
- "An Homotopy Algorithm for the Lasso with Online Observations." hom_lasso_NIPS08 [Link](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf)


---
---



