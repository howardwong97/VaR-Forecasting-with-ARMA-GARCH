import numpy as np
from scipy.optimize import minimize


def arma(phi, theta, X):
    T = len(X)
    p = len(phi)
    q = len(theta)
    epsilon = np.zeros(T)

    for t in range(T):
        if t <= (p-1):
            epsilon[t] = X[t] - X.mean()
        else:
            ar_val = np.sum([phi[i] if (i == 0) else phi[i] * X[t-i] for i in range(p)])
            ma_val = np.sum([theta[i] * epsilon[t-i-1] for i in range(q)])  # starts from theta1
            epsilon[t] = X[t] - ar_val - ma_val
    return epsilon


def garch(alpha0, alpha1, beta1, epsilon):
    T = len(epsilon)
    sigma_2 = np.zeros(T)

    for t in range(T):
        if t == 0:
            sigma_2[t] = alpha0 / (1 - alpha1 - beta1)  # initialize as unconditional variance
        else:
            sigma_2[t] = alpha0 + alpha1 * epsilon[t - 1] ** 2 + beta1 * sigma_2[t - 1]

    return sigma_2
