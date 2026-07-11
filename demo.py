import numpy as np

from src.engine import LinearRegression
from src.visualization import animate_training

g = np.random.default_rng(101)

# Generate dataset
X = g.uniform(0, 10, size=(100, 1))

true_w = 4.0
true_b = 5.0

noise = g.normal(0, 1, size=(100, 1))

y = true_w * X + true_b + noise

# Train model
model = LinearRegression(
    epochs=1000,
    learning_rate=0.01,
)

model.fit(X, y)

# Create animation
animate_training(
    X,
    y,
    model,
    true_w=true_w,
    true_b=true_b,
    filename="images/gradient_descent.gif",
)