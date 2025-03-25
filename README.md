# Project 1 

Your objective is to implement the LASSO regularized regression model using the Homotopy Method. You can read about this method in [this](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf) paper and the references therein. You are required to write a README for your project. Please describe how to run the code in your project *in your README*. Including some usage examples would be an excellent idea. You may use Numpy/Scipy, but you may not use built-in models from, e.g. SciKit Learn. This implementation must be done from first principles. You may use SciKit Learn as a source of test data.

You should create a virtual environment and install the packages in the requirements.txt in your virtual environment. You can read more about virtual environments [here](https://docs.python.org/3/library/venv.html). Once you've installed PyTest, you can run the `pytest` CLI command *from the tests* directory. I would encourage you to add your own tests as you go to ensure that your model is working as a LASSO model should (Hint: What should happen when you feed it highly collinear data?)

In order to turn your project in: Create a fork of this repository, fill out the relevant model classes with the correct logic. Please write a number of tests that ensure that your LASSO model is working correctly. It should produce a sparse solution in cases where there is collinear training data. You may check small test sets into GitHub, but if you have a larger one (over, say 20MB), please let us know and we will find an alternative solution. In order for us to consider your project, you *must* open a pull request on this repo. This is how we consider your project is "turned in" and thus we will use the datetime of your pull request as your submission time. If you fail to do this, we will grade your project as if it is late, and your grade will reflect this as detailed on the course syllabus. 

You may include Jupyter notebooks as visualizations or to help explain what your model does, but you will be graded on whether your model is correctly implemented in the model class files and whether we feel your test coverage is adequate. We may award bonus points for compelling improvements/explanations above and beyond the assignment.

Put your README here. Answer the following questions.

* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? 
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?




------
---

# LASSO Regression using the Homotopy Algorithm

## Introduction
This project implements the **LASSO (Least Absolute Shrinkage and Selection Operator) regression model** using the **Homotopy Method** from first principles. LASSO is a powerful technique for regression analysis that introduces an L1 regularization term to promote sparsity in the model parameters. The Homotopy algorithm efficiently finds solutions to the LASSO problem as the regularization parameter varies.

This project does **not** use built-in LASSO implementations from libraries like Scikit-Learn but instead builds the model from scratch using fundamental numerical computing tools such as **NumPy and SciPy**.

---

## What is LASSO Regression?
LASSO (Least Absolute Shrinkage and Selection Operator) regression is a type of linear regression that includes an L1 penalty on the absolute values of the regression coefficients. The main advantages of LASSO include:

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
The **Homotopy Algorithm** is an efficient method for solving the LASSO problem by tracking the solution path as the regularization parameter \( \lambda \) changes. Instead of solving the LASSO problem independently for multiple values of \( \lambda \), the Homotopy algorithm exploits the structure of the optimization problem to compute an entire path of solutions in an efficient manner.

In simple terms we update regularization parameter and then compute for t=0 to t=1 using updated param.
![name](https://github.com/user-attachments/assets/28038d83-85a4-4b8c-9017-5d03fc35dc31)


Key advantages of the Homotopy Algorithm:
- **Computational Efficiency**: Faster than solving multiple LASSO problems independently.
- **Path Tracking**: Provides a complete solution trajectory for different values of \( \lambda \).
- **Optimal for Sparse Solutions**: Particularly useful when the expected solution is sparse.

For further theoretical background, refer to the uploaded research paper:[Link](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf)

---

## Installation and Setup
### 1. Create a Virtual Environment
To ensure dependency management, create a virtual environment:
```sh
python -m venv lasso_env
source lasso_env/bin/activate   # On Windows use: lasso_env\Scripts\activate
```

### 2. Install Dependencies
Ensure that you have all required dependencies installed by running:
```sh
pip install -r requirements.txt
```

---

## Running the Project
### 1. Training the LASSO Model
Run the following command to train the LASSO model using the Homotopy method:
```sh
python train.py --input data.csv --lambda 0.1
```
where:
- `data.csv` is the input dataset.
- `--lambda` is the regularization parameter (default: `0.1`).

### 2. Evaluating the Model
To evaluate the trained LASSO model, run:
```sh
python evaluate.py --test test_data.csv
```
where `test_data.csv` contains test samples.

### 3. Running Tests
To validate the correctness of the implementation, run:
```sh
pytest tests/
```

---

## Answers to Required Questions
### 1. **What does the model do and when should it be used?**
The model implements **LASSO regression using the Homotopy Algorithm**. It is particularly useful for:
- **Feature Selection**: Eliminates irrelevant features by setting some coefficients to zero.
- **Handling Multicollinearity**: Works well in situations where predictors are highly correlated.
- **Sparse Models**: Reduces model complexity, making it interpretable and efficient.
- **High-Dimensional Data**: When the number of features is much larger than the number of observations.

### 2. **How was the model tested for correctness?**
- **Unit Tests**: Using `pytest`, we wrote tests to validate:
  - Correct computation of cost function.
  - Proper coefficient shrinkage for different values of \( \lambda \).
  - The Homotopy path follows expected trajectories.
- **Synthetic Data**: Generated datasets where the expected outcome was known.
- **Comparison with Scikit-Learn**: Compared the results against `sklearn.linear_model.Lasso` to verify correctness.

### 3. **Exposed Parameters for Tuning Performance**
- `lambda (\lambda)`: Controls the degree of regularization.
- `tolerance`: Convergence threshold for optimization.
- `max_iter`: Limits the number of iterations for efficiency.

### 4. **Challenges and Potential Improvements**
- **Handling Extremely Large Datasets**: Current implementation may struggle with very large feature spaces. **Possible fix:** Use batch processing or sparse matrix optimizations.
- **Highly Collinear Features**: While LASSO selects one feature among correlated ones, it may be unstable. **Possible fix:** Combine LASSO with Ridge regression (**Elastic Net**).
- **Path Efficiency**: Computing the full Homotopy path may be expensive. **Possible fix:** Use an adaptive approach to select relevant \( \lambda \) values dynamically.

---

## Conclusion
This project successfully implements LASSO regression using the Homotopy Algorithm. The model efficiently tracks the solution path as the regularization parameter varies, making it a powerful tool for feature selection and sparse modeling. By testing on synthetic and real datasets, we ensure correctness and demonstrate practical applications of LASSO regression.


---

## References
- "An Homotopy Algorithm for the Lasso with Online Observations." hom_lasso_NIPS08 [Link](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf)


---



