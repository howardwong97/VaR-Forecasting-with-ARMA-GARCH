from scipy import stats
from numpy import log, argmax
from pandas import DataFrame


def get_distribution_params(x, distribution: str) -> tuple:
    dist = getattr(stats, distribution)
    params = dist.fit(x)
    return params


def log_likelihood(x, distribution: str, return_params=True):
    dist = getattr(stats, distribution)
    params = dist.fit(x)
    if return_params:
        return dist.logpdf(x, *params).sum(), params
    else:
        return dist.logpdf(x, *params).sum()


def get_aic(k: int, llh: float) -> float:
    return 2 * (k - log(llh))


def get_best_dist(x, distributions: list, return_df=False):
    all_params, all_llh, all_aic, all_k = [], [], [], []
    for dist in distributions:
        params, llh = log_likelihood(x, dist)
        k = len(params)
        all_k.append(k)
        all_llh.append(llh)
        all_params.append(params)
        all_aic.append(get_aic(k, llh))

    if return_df:
        df = DataFrame({
            'distribution': distributions,
            'k': all_k,
            'log_likelihood': all_llh,
            'aic': all_aic
        }).set_index('distribution')
        return df

    else:
        idx = argmax(all_aic)
        return distributions[idx], all_params[idx]
