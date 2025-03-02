# Gradient Descent for Linear Regression

## Introduction to Gradient Descent

Gradient Descent is an optimization algorithm widely used in machine learning to minimize a cost function. It works by iteratively adjusting the parameters of a model in the direction that reduces the error between predicted and actual values. Imagine you're trying to find the lowest point in a valley while blindfolded—you take small steps downhill based on the slope under your feet. That’s essentially what gradient descent does!

In the context of linear regression, gradient descent finds the optimal slope (`m`) and intercept (`c`) of a line (`y = mx + c`) that best fits a set of data points. It does this by minimizing the Mean Squared Error (MSE), which measures the average squared difference between predicted and actual values.

The algorithm follows these steps:
- Initialize Parameters: Start with initial guesses for `m` and `c` (e.g., `m = 0.1`, `c = mean of target variable`).
- Compute Predictions: Use the current `m` and `c` to predict outputs for given inputs.
- Calculate Cost: Compute the MSE to quantify how far off the predictions are.
- Compute Gradients: Calculate the partial derivatives of the cost function with respect to `m` and `c` to determine the direction and magnitude of the steepest increase.
- Update Parameters: Adjust `m` and `c` in the opposite direction of the gradients, scaled by a learning rate (`alpha`), to reduce the cost.
- Repeat: Iterate until the cost converges (stops changing significantly) or a maximum number of iterations is reached.

This repository contains a Python implementation of gradient descent applied to a housing dataset, predicting sale prices based on flat area.

## About This Project

This project demonstrates a basic gradient descent algorithm for linear regression. It uses a small sample of housing data (`Transformed_Housing_Data2.csv`) with two features: `Sale_Price` (target) and `Flat Area (in Sqft)` (predictor). The code:
- Loads and preprocesses the data using standardization (`StandardScaler`).
- Implements gradient descent from scratch with helper functions for initialization, prediction, cost computation, gradient calculation, and parameter updates.
- Visualizes the final regression line using Matplotlib.

### Key Features
- Dataset: A sample of 30 houses with sale prices and flat areas.
- Preprocessing: Features are standardized to have zero mean and unit variance.
- Convergence: Stops when the cost change is less than `10^-7` or after 1000 iterations.
- Visualization: Plots the data points and the fitted regression line.

## Code Structure

- `param_init(Y)`: Initializes slope `m` and intercept `c`.
- `generate_pred(m, c, X)`: Generates predictions using `y = mx + c`.
- `comput_cost(prediction, Y)`: Computes the Mean Squared Error.
- `gradients(prediction, Y, X)`: Calculates gradients for `m` and `c`.
- `param_update(m_old, c_old, Gm_old, Gc_old, alpha)`: Updates parameters using the learning rate.
- `result(m, c, X, Y, cost, predictions, i)`: Prints convergence info and plots the result.
- Main Loop: Runs gradient descent with a learning rate of `alpha = 0.01`.

## Requirements

To run this code, you'll need the following Python libraries:
- pandas
- numpy
- matplotlib
- scikit-learn

Install them using pip:
```bash
pip install pandas numpy matplotlib scikit-learn
