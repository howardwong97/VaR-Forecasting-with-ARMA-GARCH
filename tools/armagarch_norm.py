import numpy as np
import itertools
from scipy.optimize import minimize
import warnings
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def get_epsilon(c, phi, theta, r):
    T = len(r)
    eps = np.zeros(T)
    for t in range(T):
        if t < len(phi):
            eps[t] = r[t] - np.mean(r)
        else:
            ar_component = np.sum(np.array([phi[i] * r[t - 1 - i] for i in range(len(phi))], dtype=np.float64))
            ma_component = np.sum(np.array([theta[i] * eps[t - 1 - i] for i in range(len(theta))], dtype=np.float64))
            eps[t] = r[t] - c - ar_component - ma_component
    return eps


def get_sigma2(omega, alpha, beta, gamma, r, eps, gjr=False):
    T = len(eps)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, T):
        if gjr:
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + gamma * r[t - 1] ** 2 * (eps[t - 1] < 0) + beta * sigma2[
                t - 1]
        else:
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def get_loglikelihood(params, p, q, r, gjr=False, neg=True, out=None):
    c = params[0]
    phi = params[1:p + 1]
    theta = params[p + 1:p + q + 1]

    eps = get_epsilon(c, phi, theta, r)

    if gjr:
        omega, alpha, gamma, beta = params[-4:]
    else:
        omega, alpha, beta = params[-3:]
        gamma = None
    sigma2 = get_sigma2(omega, alpha, beta, gamma, r, eps, gjr)

    llh = - 0.5 * (np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)
    if neg:
        llh = -llh
    total_llh = np.sum(llh)

    if out is None:
        return total_llh

    else:
        return total_llh, llh, sigma2


def gjr_constraint(params, p=1, q=1, r=1, gjr=False, neg=True, out=None):
    return 0.99999999999999999 - params[-3] - params[-2] / 2 - params[-1]


def non_gjr_constraint(params, p=1, q=1, r=1, gjr=False, neg=True, out=None):
    return 0.99999999999999999 - params[-2] - params[-1]


def order_determination(r, max_p, max_q, gjr=False, verbose=False):
    np.seterr(divide='ignore', invalid='ignore', over='ignore')  #
    order_combinations = list(itertools.product(np.arange(max_p + 1), np.arange(max_q + 1)))
    best_aic = np.inf

    for order in tqdm(order_combinations):
        p, q = order[0], order[1]
        bounds = [(-10 * np.abs(np.mean(r)), 10 * np.abs(np.mean(r)))] + [(-0.99999, 0.99999) for _ in range(p + q)]
        bounds += [(np.finfo(np.float64).eps, 2 * np.var(r))]
        alpha_bounds, gamma_bounds, beta_bounds = [(np.finfo(np.float64).eps, 0.999999999) for _ in range(3)]

        if gjr:
            bounds = bounds + [alpha_bounds] + [gamma_bounds] + [beta_bounds]
            initial_params = [0.001 for _ in range(p + q + 1)]
            initial_params += [0.001, 0.1, 0.01, 0.8]
            initial_params = np.array(initial_params).reshape((len(initial_params),))
            con = [
                {'type': 'ineq', 'fun': gjr_constraint},
                {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x[1:p + 1]) - np.finfo(np.float64).eps}
            ]
        else:
            bounds = bounds + [alpha_bounds] + [beta_bounds]
            initial_params = [0.001 for _ in range(p + q + 1)]
            initial_params += [0.001, 0.1, 0.8]
            initial_params = np.array(initial_params).reshape((len(initial_params),))
            con = [
                {'type': 'ineq', 'fun': non_gjr_constraint},
                {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x[1:p + 1]) - np.finfo(np.float64).eps}
            ]

        args = (p, q, r, gjr, True, None)
        res = minimize(get_loglikelihood, initial_params, args=args, bounds=bounds, constraints=con, method='SLSQP')
        current_aic = 2 * res.fun + 2 * len(initial_params)

        if current_aic < best_aic:
            best_aic = current_aic
            best_p, best_q = p, q
            best_params = res.x
            if verbose:
                if gjr:
                    print('Current best: ARMA({},{})-GJR-GARCH(1,1,1), AIC = {}'.format(str(p), str(q), current_aic))
                else:
                    print('Current best: ARMA({},{})-GARCH(1,1), AIC = {}'.format(str(p), str(q), current_aic))

    if verbose:
        print('Order determination complete for ARMA(p,q).')
        print('p =', best_p)
        print('q =', best_q)
        print('AIC =', best_aic)
    return best_p, best_q, best_params