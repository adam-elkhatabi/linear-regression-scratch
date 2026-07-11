# Linear Regression from Scratch (NumPy)

A pure **NumPy** implementation of Linear Regression trained with **Batch Gradient Descent**. This project was built to understand the mathematics behind linear regression rather than relying on machine learning libraries such as scikit-learn.

---

## Features

* ✅ Linear Regression implemented from scratch
* ✅ Vectorized implementation using NumPy
* ✅ Batch Gradient Descent optimization
* ✅ Mean Squared Error (MSE) loss
* ✅ Support for multiple input features
* ✅ Training loss tracking
* ✅ Animated visualization of gradient descent
* ✅ Clean, modular project structure

---

## Mathematical Model

The prediction function is

[
\hat{y} = XW + b
]

where:

* (X) is the feature matrix
* (W) is the weight vector
* (b) is the bias
* (\hat{y}) is the predicted value

The model minimizes the Mean Squared Error:

[
J(W,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}_i-y_i)^2
]

The gradients are

[
\frac{\partial J}{\partial W}
=============================

\frac{1}{m}X^T(\hat{y}-y)
]

[
\frac{\partial J}{\partial b}
=============================

\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)
]

The parameters are updated using Gradient Descent:

[
W := W-\alpha\frac{\partial J}{\partial W}
]

[
b := b-\alpha\frac{\partial J}{\partial b}
]

---

## Example

```python
from src.linear_regression import LinearRegression

model = LinearRegression(
    epochs=1000,
    learning_rate=0.01
)

model.fit(X, y)

predictions = model.predict(X)
```

---

## Gradient Descent Visualization

The animation below shows how Gradient Descent gradually updates the regression line until it converges to the true relationship between the data and the target.

![Gradient Descent](images/gradient_descent.gif)

---

## Example Dataset

Synthetic data is generated using

```python
X = rng.uniform(0, 10, size=(100, 1))

true_w = 4.0
true_b = 5.0

noise = rng.normal(0, 1, size=(100, 1))

y = X * true_w + true_b + noise
```

---

## Technologies

* Python
* NumPy
* Matplotlib

---

## Learning Goals

This project was created to develop a deep understanding of:

* Vectorized linear algebra
* Matrix calculus
* Gradient Descent
* Mean Squared Error
* Data normalization
* Writing clean, modular machine learning code
* Building machine learning algorithms from scratch

---

## Future Improvements

* [ ] Mini-Batch Gradient Descent
* [ ] Stochastic Gradient Descent (SGD)
* [ ] L2 Regularization (Ridge Regression)
* [ ] L1 Regularization (Lasso)
* [ ] Early Stopping
* [ ] Feature Scaling utilities
* [ ] Model serialization
* [ ] Unit tests
* [ ] Performance benchmarks
* [ ] Logistic Regression from scratch

---

## License

This project is open source and available under the MIT License.
