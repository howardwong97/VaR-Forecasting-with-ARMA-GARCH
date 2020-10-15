import numpy as np
from yahooquery import Ticker
import armagarch as ag
import multiprocessing as mp

price = Ticker('AAPL').history(period='10y', interval='1d')
price.reset_index(inplace=True)
price.drop('symbol', axis=1, inplace=True)
price.set_index('date', inplace=True)

log_returns = np.log(price['adjclose'] / price['adjclose'].shift(1)).dropna()
T = len(log_returns)

intervals = [(i - 500, i) for i in range(500, T - 1)]


def one_step_var(interval):
    t1, t2 = interval
    X = log_returns.values[t1:t2] * 100.0
    prediction_date = log_returns.index[t2 + 1].strftime('%Y-%m-%d')

    results = ag.fit_model(X, 'norm', 1, 1, True)
    r_pred, sigma2_pred = ag.one_step_prediction(X, results[0], p=1, q=1)
    r_pred = r_pred * 0.01
    sigma2_pred = sigma2_pred * 0.01 ** 2
    value_at_risk95 = ag.VaR('norm', r_pred, np.sqrt(sigma2_pred), 1 - 0.95)

    with open(
            '/home/howardwong/Desktop/Research/ARMA-GARCH-Models/data/var-forecasts/{}.txt'.format(prediction_date), 'w'
    ) as f:
        f.write(prediction_date + ',' + str(value_at_risk95) + ',' + str(log_returns.values[t2 + 1]))
        f.close()


if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    pool.map(one_step_var, intervals)
