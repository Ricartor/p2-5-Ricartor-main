[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/j6Hnhn72)
# Logistic Regression Implementation Exercise

In this exercise, you will implement a Logistic Regression classifier with different regularization options. You need to complete several key functions in the `LogisticRegressor` class.

## Functions to Complete

1. `fit` method:
    - Initialize `m` (number of examples) and `n` (number of features) from input X
    - Initialize `self.weights` as zeros with appropriate shape
    - Complete the gradient descent loop by implementing forward propagation and loss computation
    - Calculate gradients `dw` and `db`

2. `predict_proba` method:
    - Calculate logits `z` using weights and bias
    - Return probabilities using sigmoid function

3. `predict` method:
    - Convert probabilities to binary classifications using the threshold

4. `sigmoid` method:
    - Implement the sigmoid function

5. `log_likelihood` method:
    - Implement the binary cross-entropy loss function

6. Regularization methods:
    - `lasso_regularization`: Implement L1 regularization gradient
    - `ridge_regularization`: Implement L2 regularization gradient
    - `elasticnet_regularization`: Combine L1 and L2 regularization

## Testing Your Implementation

The provided test file `test_logistic_regressor.py` contains unit tests for each function. Your implementation should pass all tests. Key test cases include:

- Basic model fitting without regularization
- Probability predictions in range [0,1]
- Binary class predictions
- Correct regularization effects
- Sigmoid function behavior
- Loss calculation

## Tips
- Start with the basic functions (sigmoid, predict_proba) before tackling regularization
- Make sure you understand the mathematical formulas for each regularization type
- Use numpy for efficient array operations
- Pay attention to the shapes of your arrays

Good luck with your implementation!