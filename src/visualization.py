import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_training(
    X,
    y,
    model,
    true_w=None,
    true_b=None,
    filename="gradient_descent.gif",
):
    """
    Creates a GIF showing how the regression line converges
    during gradient descent.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 1)
        Training feature.

    y : ndarray of shape (n_samples, 1)
        Target values.

    model : LinearRegression
        Trained model containing:
            - w_history
            - b_history
            - losses

    true_w : float or None
        True weight used to generate the data.

    true_b : float or None
        True bias used to generate the data.

    filename : str
        Output gif filename.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(X, y, color="steelblue", alpha=0.7, label="Training Data")

    # x values used to draw the line
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    # Draw true regression line
    if true_w is not None and true_b is not None:
        y_true = x_line * true_w + true_b
        ax.plot(
            x_line,
            y_true,
            color="black",
            linewidth=3,
            label="True Line",
        )

    # Model line
    (model_line,) = ax.plot(
        [],
        [],
        color="crimson",
        linewidth=2.5,
        label="Model",
    )

    title = ax.set_title("")

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(y.min() - 2, y.max() + 2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.legend()

    def update(frame):

        w = model.w_history[frame]
        b = model.b_history[frame]

        y_pred = x_line @ w + b

        model_line.set_data(x_line, y_pred)

        title.set_text(
            f"Epoch: {frame}    Loss: {model.losses[frame]:.4f}"
        )

        return model_line, title

    frames = [
    0,
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    999,
    999,
    999,
    999]

    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=30,
        blit=True,
    )

    animation.save(
        filename,
        writer="pillow",
        fps=2,
    )

    plt.close()

    print(f"Saved animation to '{filename}'")