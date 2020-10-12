import itertools
import warnings

import numpy as np
from scipy import optimize
from scipy import stats
from scipy.special import gamma as GammaFunc
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


def one_step_prediction(r, estimates, p, q, gjr=False):
    c = estimates[0]
    phi = estimates[1:p + 1]
    theta = estimates[p + 1:p + q + 1]
    if gjr:
        omega, alpha, gamma, beta = estimates[-4:]
    else:
        omega, alpha, beta = estimates[-3:]
        gamma = None
    eps = get_epsilon(c, phi, theta, r)
    sigma2 = get_sigma2(omega, alpha, beta, gamma, r, eps, gjr)

    r_pred = c + np.sum([phi[i] * r[-i] for i in range(p)]) + np.sum([theta[j] * eps[-j] for j in range(p)])
    sigma2_pred = omega + alpha * eps[-1] ** 2 + beta * sigma2[-1]
    if gjr:
        sigma2_pred += gamma * r[-1] ** 2 * (eps[-1] < 0)
    return r_pred, sigma2_pred


def VaR(dist, mu, sigma, threshold, df=None):
    if dist in ['norm', 'normal']:
        return stats.norm.ppf(threshold, mu, sigma)
    elif dist == 't':
        return stats.t.ppf(threshold, df, mu, sigma)


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


def cons0_norm(params, p, q, r, gjr=False, out=None):
    if gjr:
        alpha, gamma, beta = params[-3:]
        return 1.0 - np.finfo(np.float64).eps - alpha - gamma / 2 - beta
    else:
        alpha, beta = params[-2:]
        return 1.0 - np.finfo(np.float64).eps - alpha - beta


def cons1_norm(params, p, q, r, gjr=False, out=None):
    return 1 - np.sum(params[1:p + 1]) - np.finfo(np.float64).eps


def cons0_t(params, p, q, r, gjr=False, out=None):
    if gjr:
        alpha, gamma, beta = params[-4], params[-3], params[-2]
        return 1.0 - np.finfo(np.float64).eps - alpha - gamma / 2 - beta
    else:
        alpha, beta = params[-3], params[-2]
        return 1.0 - np.finfo(np.float64).eps - alpha - beta


def cons1_t(params, p, q, r, gjr=False, out=None):
    return 1 - np.sum(params[1:p + 1]) - np.finfo(np.float64).eps


def consv_eq(params, p, q, r, gjr=False, out=None):
    v = params[-1]
    return max([v - int(v)])


def consv_ieq(params, p, q, r, gjr=False, out=None):
    v = params[-1]
    return v - 3


def fit_model(r, dist: str, p: int, q: int, gjr=False):
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    e = np.finfo(np.float64).eps
    bounds = [(-10 * np.abs(np.mean(r)), 10 * np.abs(np.mean(r)))] + \
             [(-0.9999999999, 0.9999999999) for _ in range(p + q)] + \
             [(e, 2 * np.var(r))]
    alpha_bounds, gamma_bounds, beta_bounds = [(e, 1.0 - e) for _ in range(3)]
    initial_params = [0.001 for _ in range(p + q + 1)]

    if dist == 't':
        min_func = t_negative_llh
        v_bounds = (3, None)
        eqcons = [consv_eq]
        ieqcons = [cons0_t, cons1_t, consv_ieq]
        if gjr:
            initial_params = initial_params + [0.001, 0.1, 0.01, 0.8, 3]
            bounds = bounds + [alpha_bounds, gamma_bounds, beta_bounds, v_bounds]
        else:
            initial_params = initial_params + [0.001, 0.1, 0.8, 3]
            bounds = bounds + [alpha_bounds, beta_bounds, v_bounds]

    elif dist in ['normal', 'norm']:
        min_func = norm_negative_llh
        eqcons = []
        ieqcons = [cons0_norm, cons1_norm]
        if gjr:
            initial_params = initial_params + [0.001, 0.1, 0.01, 0.8]
            bounds = bounds + [alpha_bounds, gamma_bounds, beta_bounds]
        else:
            initial_params = initial_params + [0.001, 0.1, 0.8]
            bounds = bounds + [alpha_bounds, beta_bounds]

    result = optimize.fmin_slsqp(
        func=min_func,
        x0=initial_params,
        ieqcons=ieqcons,
        eqcons=eqcons,
        bounds=bounds,
        epsilon=1e-6,
        acc=1e-7,
        full_output=True,
        iprint=0,
        args=(p, q, r, gjr, None),
        iter=300
    )

    return result


def order_determination(r, dist: str, max_p: int, max_q: int, gjr=False, verbose=False):
    order_combinations = list(itertools.product(np.arange(max_p + 1), np.arange(max_q + 1)))
    best_aic = np.inf
    for order in tqdm(order_combinations):
        p, q = order[0], order[1]
        result = fit_model(r, dist, p, q, gjr=gjr)
        current_aic = 2 * result[1] + 2 * len(result[0])
        if current_aic < best_aic:
            best_aic = current_aic
            best_p, best_q = p, q
            best_params = result[0]
            if verbose:
                if gjr:
                    print('Current best: ARMA({},{})-GJR-GARCH(1,1,1), AIC = {}'.format(str(p), str(q), current_aic))
                else:
                    print('Current best: ARMA({},{})-GARCH(1,1), AIC = {}'.format(str(p), str(q), current_aic))
    if verbose:
        print('Order determination complete with p = {} and q = {}'.format(str(best_p), str(best_q)))
        print('AIC =', best_aic)
    return best_params, best_p, best_q


def hessian_2sided(fun, theta, args):
    f = fun(theta, *args)
    h = 1e-5 * np.abs(theta)
    thetah = theta + h
    h = thetah - theta
    K = np.size(theta, 0)
    h = np.diag(h)

    fp = np.zeros(K)
    fm = np.zeros(K)
    for i in range(K):
        fp[i] = fun(theta + h[i], *args)
        fm[i] = fun(theta - h[i], *args)

    fpp = np.zeros((K, K))
    fmm = np.zeros((K, K))
    for i in range(K):
        for j in range(i, K):
            fpp[i, j] = fun(theta + h[i] + h[j], *args)
            fpp[j, i] = fpp[i, j]
            fmm[i, j] = fun(theta - h[i] - h[j], *args)
            fmm[j, i] = fmm[i, j]

    hh = (np.diag(h))
    hh = hh.reshape((K, 1))
    hh = hh @ hh.T

    H = np.zeros((K, K))
    for i in range(K):
        for j in range(i, K):
            H[i, j] = (fpp[i, j] - fp[i] - fp[j] + f
                       + f - fm[i] - fm[j] + fmm[i, j]) / hh[i, j] / 2
            H[j, i] = H[i, j]

    return H


def get_summary_stats(X, estimates, dist: str, p: int, q: int, gjr=False, print_output=False):
    step = 1e-5 * estimates
    T = len(X)
    scores = np.zeros((T, len(estimates)))

    for i in range(len(estimates)):
        h = step[i]
        delta = np.zeros(len(estimates))
        delta[i] = h
        if dist in ['norm', 'normal']:
            _, llh_neg, _ = norm_negative_llh(estimates - delta, p, q, X, gjr=gjr, out=True)
            _, llh_pos, _ = norm_negative_llh(estimates + delta, p, q, X, gjr=gjr, out=True)
        elif dist == 't':
            _, llh_neg, _ = t_negative_llh(estimates - delta, p, q, X, gjr=gjr, out=True)
            _, llh_pos, _ = t_negative_llh(estimates + delta, p, q, X, gjr=gjr, out=True)
        scores[:, i] = (llh_pos - llh_neg) / (2 * h)
    V = (scores.T @ scores) / T

    if dist in ['norm', 'normal']:
        J = hessian_2sided(norm_negative_llh, estimates, (p, q, X, gjr)) / T
    elif dist == 't':
        J = hessian_2sided(t_negative_llh, estimates, (p, q, X, gjr)) / T
    Jinv = np.mat(np.linalg.inv(J))

    asymptotic_variance = np.asarray(Jinv * np.mat(V) * Jinv / T)
    std_err = np.sqrt(np.diag(asymptotic_variance))
    tstats = np.abs(estimates / std_err)
    pvals = [stats.t.sf(np.abs(i), T - 1) * 2 for i in tstats]
    output = np.vstack((estimates, std_err, tstats, pvals)).T

    if print_output:
        print('Parameter   Estimate       Std. Err.      T-stat     p-value')
        param = ['c'] + ['phi{}'.format(i) for i in range(p)] + \
                ['theta{}'.format(i) for i in range(q)] + \
                ['omega', 'alpha']
        if gjr:
            param += ['gamma', 'beta']
        else:
            param += ['beta']
        if dist == 't':
            param = param + ['v']
        for i in range(len(param)):
            print('{0:<11} {1:>0.6f}        {2:0.6f}    {3: 0.5f}    {4: 0.5f}'.format(
                param[i], output[i, 0], output[i, 1], output[i, 2], output[i, 3])
            )
    return estimates, std_err, tstats, pvals
