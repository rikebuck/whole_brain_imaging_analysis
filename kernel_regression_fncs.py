import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import vonmises, norm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import numpy as np

import numpy as np
from scipy.stats import vonmises, norm

def circular_distance(theta1, theta2):
    return np.angle(np.exp(1j * (theta1 - theta2)))

def kernel_regression_2d(
    X_train, y_train, X_eval,
    h_x=0.3, h_y=0.3,
    circular_x=True,
    kernel_type=("vonmises", "gaussian"),  # or ("uniform", "uniform")
    return_std=False
):
    x, y = X_train[:, 0], X_train[:, 1]
    x_eval, y_eval = X_eval[:, 0], X_eval[:, 1]
    z_est = np.zeros(len(x_eval))
    z_std = np.zeros(len(x_eval))

    for i in range(len(x_eval)):


        if kernel_type[0] == "vonmises" and circular_x:
            wx = vonmises.pdf(circular_distance(x, x_eval[i]), kappa=1/h_x)
        elif kernel_type[0] == "gaussian":
            wx = norm.pdf((x - x_eval[i]) / h_x)
        elif kernel_type[0] == "uniform":
            angle_diff = circular_distance(x, x_eval[i])
            wx = ((np.abs(angle_diff) <= h_x).astype(float)) / (2 * h_x)
        else:
            raise ValueError(f"Unsupported y kernel type: {kernel_type[1]}")

        # Y-axis kernel
        if kernel_type[1] == "gaussian":
            wy = norm.pdf((y - y_eval[i]) / h_y)
        elif kernel_type[1] == "uniform":
            wy = ((np.abs(y - y_eval[i]) <= h_y).astype(float)) / (2 * h_y)
        else:
            raise ValueError(f"Unsupported y kernel type: {kernel_type[1]}")

        weights = wx * wy
        w_sum = weights.sum()

        if w_sum > 0:
            weights /= w_sum
            z_est[i] = np.sum(weights * y_train)
            if return_std:
                var = np.sum(weights * (y_train - z_est[i])**2)
                n_eff = 1.0 / np.sum(weights**2)
                z_std[i] = np.sqrt(var) / np.sqrt(n_eff)
        else:
            z_est[i] = np.nan
            if return_std:
                z_std[i] = np.nan

    return (z_est, z_std) if return_std else z_est


def fit_and_evaluate_kernel_regression(
    X,
    y,
    method="kernel",
    plot=True,
    ax=None,
    bandwidths=(0.3, 0.3),
    kernel_type=("vonmises", "gaussian"),
    test_size=0.2,
    random_state=0,
    plot_points=100, alpha=1, return_est = False
):
    X = np.atleast_2d(X)
    if X.shape[0] != len(y):
        X = X.T

    if X.shape[1] != 2:
        raise ValueError("This function assumes X has shape (n_samples, 2)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if method == "logistic":
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

    elif method == "kernel":
        h_x, h_y = bandwidths
        y_prob, _ = kernel_regression_2d(
            X_train, y_train, X_test,
            h_x=h_x, h_y=h_y, kernel_type=kernel_type, return_std=True
        )

    else:
        raise ValueError("Method must be 'logistic' or 'kernel'")

    y_pred = (y_prob >= 0.5).astype(int)

    try:
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob, labels=[0, 1])
        auc = roc_auc_score(y_test, y_prob)
    except:
        pass

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        if method == "logistic":
            # Plot prediction as function of βᵀx (decision axis)
            coefs = model.coef_[0]
            intercept = model.intercept_[0]

            # Project training inputs onto decision axis
            proj_train = X_train @ coefs + intercept
            proj_grid = np.linspace(proj_train.min()-4, proj_train.max()+4, plot_points)
            X_eval = np.outer(np.ones_like(proj_grid), coefs)
            X_eval = proj_grid[:, None] * coefs / np.linalg.norm(coefs)**2
            X_eval = X_eval  # already in direction of coefs, rescaled
            z_vals = X_eval @ coefs + intercept
            y_plot = 1 / (1 + np.exp(-z_vals))

            ax.plot(z_vals, y_plot, label="Logistic Regression", color="C1")
            z_test = X @ coefs + intercept
            ax.scatter(z_test, y, c='k', alpha=0.3, s=20, label="Test Labels")
            ax.set_xlabel("β₀ + β₁x₀ + β₂x₁")

        elif method == "kernel":
            # Plot prediction as function of x0, with x1 fixed at mean
            x0_grid = np.linspace(-np.pi, np.pi, plot_points)
            x1_mean = np.mean(X_train[:, 1])
            X_eval = np.column_stack([x0_grid, np.full_like(x0_grid, x1_mean)])
            y_plot, y_sem = kernel_regression_2d(
                X_train, y_train, X_eval,
                h_x=bandwidths[0], h_y=bandwidths[1],
                kernel_type=kernel_type, return_std=True
            )
            # y_sem *= 1.96
            ax.plot(x0_grid, y_plot, label="Kernel Regression", color="purple", alpha = alpha)
            ax.fill_between(x0_grid, y_plot - y_sem, y_plot + y_sem, color="purple", label="±1 SEM", alpha=alpha)
            ax.set_xlabel(r"$\theta$")

            # ax.scatter(X_test[:, 0], y_test, c='k', alpha=0.3, s=20, label="Test Labels")
        ax.set_ylabel("P(y=1)")
        ax.set_title(f"{method.capitalize()} Regression")
        plt.tight_layout()

    if return_est:
        x0_grid = np.linspace(-np.pi, np.pi, plot_points)
        x1_mean = np.mean(X_train[:, 1])
        X_eval = np.column_stack([x0_grid, np.full_like(x0_grid, x1_mean)])
        y_plot, y_sem = kernel_regression_2d(
                X_train, y_train, X_eval,
                h_x=bandwidths[0], h_y=bandwidths[1],
                kernel_type=kernel_type, return_std=True
            )
        return y_plot
    return acc, ll, auc