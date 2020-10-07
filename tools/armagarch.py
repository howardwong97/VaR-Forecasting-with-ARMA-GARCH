from scipy.special import gamma as GammaFunc
from scipy.optimize import minimize
from tqdm.notebook import tqdm
import numpy as np
import itertools
import warnings

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


def norm_negative_llh(params, p, q, r, gjr=False, out=None):
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
    neg_llh = -llh  # minimize negative log likelihood
    total_llh = np.sum(neg_llh)
    if out is None:
        return total_llh
    else:
        return total_llh, llh, sigma2


def t_negative_llh(params, p, q, r, gjr=False, out=None):
    c, phi, theta = params[0], params[1:p + 1], params[p + 1:p + q + 1]
    if gjr:
        omega, alpha, gamma, beta, v = params[-5:]
    else:
        omega, alpha, beta, v = params[-4:]
        gamma = None
    eps = get_epsilon(c, phi, theta, r)
    sigma2 = get_sigma2(omega, alpha, beta, gamma, r, eps, gjr)
    llh = np.log(GammaFunc((v + 1) / 2) / (GammaFunc(v / 2) * np.sqrt(np.pi * (v - 2) * sigma2)) * \
          (1 + eps ** 2 / ((v - 2) * sigma2)) ** (-(v + 1) / 2))
    neg_llh = -llh  # minimize negative log likelihood
    total_llh = np.sum(neg_llh)
    if out is None:
        return total_llh
    else:
        return total_llh, llh, sigma2


def fit_model(r, dist: str, p: int, q: int, gjr=False):
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    e = np.finfo(np.float64).eps
    bounds = [(-10 * np.abs(np.mean(r)), 10 * np.abs(np.mean(r)))] +\
             [(-0.9999999999, 0.9999999999) for _ in range(p + q)] +\
             [(e, 2 * np.var(r))]
    alpha_bounds, gamma_bounds, beta_bounds = [(e, 1.0-e) for _ in range(3)]
    initial_params = [0.001 for _ in range(p + q + 1)]

    if dist == 't':
        v_bounds = (3, None)
        con = [{'type': 'eq', 'fun': lambda x: max([x[-1]-int(x[-1])])},
               {'type': 'ineq', 'fun': lambda x: x[-1] - 3}]
        if gjr:
            initial_params = initial_params + [0.001, 0.1, 0.01, 0.8, 3]
            con = con + [{'type': 'ineq', 'fun': lambda x: 1.0 - np.finfo(np.float64).eps - x[-4] - x[-3] / 2 - x[-2]}]
            bounds = bounds + [alpha_bounds, gamma_bounds, beta_bounds, v_bounds]
        else:
            initial_params = initial_params + [0.001, 0.1, 0.8, 3]
            con = con + [{'type': 'ineq', 'fun': lambda x: 1.0 - np.finfo(np.float64).eps - x[-3] - x[-2]}]
            bounds = bounds + [alpha_bounds, beta_bounds, v_bounds]

        result = minimize(
            fun=t_negative_llh,
            x0=initial_params,
            args=(p, q, r, gjr),
            method='SLSQP',
            bounds=bounds,
            constraints=con
        )

    elif dist in ['normal', 'norm']:
        if gjr:
            initial_params = initial_params + [0.001, 0.1, 0.01, 0.8]
            con = [{'type': 'ineq', 'fun': lambda x: 1.0 - np.finfo(np.float64).eps - x[-3] - x[-2] / 2 - x[-1]},
                   {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x[1:p + 1]) - np.finfo(np.float64).eps}]
            bounds = bounds + [alpha_bounds, gamma_bounds, beta_bounds]
        else:
            initial_params = initial_params + [0.001, 0.1, 0.8]
            con = [{'type': 'ineq', 'fun': lambda x: 1.0 - np.finfo(np.float64).eps - x[-2] - x[-1]},
                   {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x[1:p + 1]) - np.finfo(np.float64).eps}]
            bounds = bounds + [alpha_bounds, beta_bounds]

        result = minimize(
            fun=norm_negative_llh,
            x0=initial_params,
            args=(p, q, r, gjr),
            method='SLSQP',
            bounds=bounds,
            constraints=con
        )

    return result


def order_determination(r, dist: str, max_p: int, max_q: int, gjr=False, verbose=False):
    if dist in ['normal', 'norm']:
        dist_text = 'Normal distribution'
    else:
        dist_text = 'Students t-distribution'
    print('Determining order for ARMA(p,q)-GJR-GARCH(1,1,1) with {}...'.format(dist_text))
    order_combinations = list(itertools.product(np.arange(max_p + 1), np.arange(max_q + 1)))
    best_aic = np.inf
    for order in tqdm(order_combinations):
        p, q = order[0], order[1]
        result = fit_model(r, dist, p, q, gjr=gjr)
        current_aic = 2 * result.fun + 2 * len(result.x)
        if current_aic < best_aic:
            best_aic = current_aic
            best_p, best_q = p, q
            best_params = result.x
            if verbose:
                if gjr:
                    print('Current best: ARMA({},{})-GJR-GARCH(1,1,1), AIC = {}'.format(str(p), str(q), current_aic))
                else:
                    print('Current best: ARMA({},{})-GARCH(1,1), AIC = {}'.format(str(p), str(q), current_aic))
    if verbose:
        print('Order determination complete with p = {} and q = {}'.format(str(best_p), str(best_q)))
        print('AIC =', best_aic)
    return best_params, best_p, best_q
