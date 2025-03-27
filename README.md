# Project 1 ( Given Problem For Project )

Your objective is to implement the LASSO regularized regression model using the Homotopy Method. You can read about this method in [this](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf) paper and the references therein. You are required to write a README for your project. Please describe how to run the code in your project *in your README*. Including some usage examples would be an excellent idea. You may use Numpy/Scipy, but you may not use built-in models from, e.g. SciKit Learn. This implementation must be done from first principles. You may use SciKit Learn as a source of test data.

You should create a virtual environment and install the packages in the requirements.txt in your virtual environment. You can read more about virtual environments [here](https://docs.python.org/3/library/venv.html). Once you've installed PyTest, you can run the `pytest` CLI command *from the tests* directory. I would encourage you to add your own tests as you go to ensure that your model is working as a LASSO model should (Hint: What should happen when you feed it highly collinear data?)

In order to turn your project in: Create a fork of this repository, fill out the relevant model classes with the correct logic. Please write a number of tests that ensure that your LASSO model is working correctly. It should produce a sparse solution in cases where there is collinear training data. You may check small test sets into GitHub, but if you have a larger one (over, say 20MB), please let us know and we will find an alternative solution. In order for us to consider your project, you *must* open a pull request on this repo. This is how we consider your project is "turned in" and thus we will use the datetime of your pull request as your submission time. If you fail to do this, we will grade your project as if it is late, and your grade will reflect this as detailed on the course syllabus. 

You may include Jupyter notebooks as visualizations or to help explain what your model does, but you will be graded on whether your model is correctly implemented in the model class files and whether we feel your test coverage is adequate. We may award bonus points for compelling improvements/explanations above and beyond the assignment.

---
---
## Team Members

  - Dhruv Bhimani (A20582831)
  - Manushi Patel (A20575366)
  - Smit Dhameliya (A20593154)
---
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

- What does the model you have implemented do and when should it be used?

    - This model, LassoHomotopyModel, implements Lasso regression using the Homotopy method, which efficiently computes the regularization path as the regularization parameter ‘mu’ changes. It is designed for 
      sparse regression problems, selecting only the most important features while shrinking others to zero.

      When to Use It?
  
    -  High-dimensional datasets where feature selection is essential.
  
    -  Streaming or online learning scenarios where new data updates the model dynamically.
  
    -  When needing efficient computation of Lasso solutions, as the Homotopy approach avoids recomputing from scratch for each regularization step.
  
    -  Interpretability-focused applications, since it provides a clear active set of selected features.


* How did you test your model to determine if it is working reasonably correctly?
  - Our testing approach ensures correctness by splitting the dataset into training, validation, and test sets. Model 1 is trained on 75% of the data and evaluated with R² Score. Model 2 implements sequential learning, updating the model iteratively with validation data before testing. Both models check for coefficient sparsity and compare learned coefficients. This setup ensures meaningful updates, and prevents excessive sparsity. The results confirm that the model adapts well while maintaining performance


* What parameters have you exposed to users of your implementation in order to tune performance?
  - our implementation currently does not expose tunable parameters; it initializes lambda (μ) dynamically as 0.1 * max(|X^T y|) and updates it during the homotopy path.
 
    
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
    1.Highly Correlated Features (Multicollinearity)

    -  When features are highly correlated, the homotopy method may struggle to select the correct variables, leading to unstable coefficient updates.
    -  Potential Fix: Implement feature decorrelation techniques or adaptive selection rules.





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
![WhatsApp Image 2025-03-26 at 22 21 50_8fb14c68](https://github.com/user-attachments/assets/f80ff952-11f1-46f1-af79-a6597303530c)

---
---

## Testing Different Datasets

We have tested different datasets including datasets with high colinearity and low colinearity. Let us observe the details of data and results.


| **Data File**  | **Details** |  **Colinear** | **Sparse Solution** | **Sparse Var**  | **Output** |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| small_test.csv  | x_1 and x_2 are the primary drivers of y (positive and negative, respectively), with x_0 having a moderate negative effect. No significant collinearity. All features are largely independent correlations < 0.15.  | No  | No | 0/3  | ![WhatsApp Image 2025-03-26 at 22 44 56_d30b27f1](https://github.com/user-attachments/assets/dbb25600-64bb-4208-b3f3-93f073f8768f) |
| collinear_data.csv  | X_4-X_8 (and X_1) dominate target’s variation positively, others have minimal impact on target. X_4 to X_8 are essentially identical, X_1 is strongly related, while X_2, X_3, X_9, X_10 are distinct.  | Highly  | Yes | 8/10  | ![WhatsApp Image 2025-03-26 at 22 46 50_413cef7e](https://github.com/user-attachments/assets/6a619e47-562d-496c-9bc0-3cd148818d0b) |
| non-correlated_data.csv  | All correlations between x1, x2, x3, and x4 are very weak  | No  | No | 0/4  | ![image](https://github.com/user-attachments/assets/41bfb2f5-3a6a-4b4f-9383-95379ca77eff) |
| test_2.csv  | x5, x3, x4, x6, and x7 are the primary drivers of y, all negatively. x2, x8, and x1 have lesser influence; x9 is nearly irrelevant linearly. x3 and x4: Perfect correlation (1.00). These are essentially identical (x4 = x3 / 2). x6 and x7: Perfect correlation (1.00). These are identical (x7 = x6 / 2). x3, x4, and x5: Very strong correlation (~0.95). x5 closely follows x3 and x4. x3, x4, x5, x6, x7: Moderate to strong correlations (~0.60-0.65), suggesting a cluster of related features. Other pairs (e.g., x1-x2, x8-x9) show weak correlations < 0.25  | Highly  | Yes | 5/9  | ![WhatsApp Image 2025-03-26 at 22 16 32_ecb6adc0](https://github.com/user-attachments/assets/1a61bdf4-86f9-49f1-8fcf-edcdc88d9293) |
| data_1.csv  | x4, x3, and x6 are the primary drivers of y. All pairwise correlations between features are weak 0.10, indicating no significant collinearity. Features x1 to x9 appear largely independent of each other. | Highly  | Yes | 7/9  | ![WhatsApp Image 2025-03-26 at 22 21 50_d0a1779e](https://github.com/user-attachments/assets/71d8d22b-2714-4728-a7c8-84262965eec0) |
| data_2.csv  | x5-x9 dominate y’s variation positively, x1/x2 oppose it strongly, and x3/x4 have a moderate positive effect.x1 and x2: Perfect correlation (1.00). Likely x2 = x1/2. x3 and x4: Perfect correlation (1.00). Likely x4 = x3/2. x5 to x9: Perfect correlation (1.00). These are identical variables. x1/x2 vs. x5-x9: Strong negative correlation (-0.90), indicating an inverse relationship. x3/x4 vs. x5-x9: Moderate positive correlation (0.51). x1/x2 vs. x3/x4: Weak correlation (0.06), suggesting independence. | Moderately  | Moderate | 4/9  | ![WhatsApp Image 2025-03-26 at 22 22 22_8b798f23](https://github.com/user-attachments/assets/79cff691-26ea-489c-9271-0eaa58181da5) |


---
---

## compare.py and generate.py


compare.py 

This code loads a dataset, extracts feature columns (X) and the target column (y), and then applies three different Lasso regression models:

homotopy_model - OUR custom Lasso model built from scratch using the Homotopy method.
lars_model - The built-in LassoLars model from Scikit-learn.
lasso_model - The built-in Lasso regression model from Scikit-learn.
All three models are trained on the dataset, and their coefficients are printed for comparison. This helps evaluate how well the custom Homotopy Lasso model performs compared to standard implementations like LassoLars and Lasso.


generate.py

This code generates a synthetic dataset with 100,000 rows and 10 columns (9 features and 1 target variable y). The features (x1 to x9) are randomly sampled from different uniform distributions. The target variable y is computed using a nonlinear formula involving additions, multiplications, and subtractions of the features. The dataset's estimated memory size in within 20MB.



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
This project successfully implements LASSO regression using the Homotopy Algorithm. The model efficiently tracks the solution path as the regularization parameter is updated dynamically, making it a powerful tool for feature selection and sparse modeling. By testing on synthetic and real datasets, we ensure correctness and demonstrate practical applications of LASSO regression.


---
---

## References
- "A Homotopy Algorithm for the Lasso with Online Observations." hom_lasso_NIPS08 [Link](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf)


---
---



