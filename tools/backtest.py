import numpy as np
from yahooquery import Ticker
import armagarch as ag
import multiprocessing as mp

price = Ticker('AAPL').history(start='2010-01-01', interval='1d')
price.reset_index(inplace=True)
price.drop('symbol', axis=1, inplace=True)
price.set_index('date', inplace=True)
price.to_csv('/home/howardwong/Desktop/Research/ARMA-GARCH-Models/data/aapl_data/aapl.csv')

log_returns = np.log(price['adjclose'] / price['adjclose'].shift(1)).dropna()
T = len(log_returns)

intervals = [(i-500, i) for i in range(500, T-1)]


def one_step_var(interval):
    t1, t2 = interval
    X = log_returns.values[t1:t2] * 100.0
    prediction_date = log_returns.index[t2+1].strftime('%Y-%m-%d')

    estimates, p, q = ag.order_determination(X, 'norm', 6, 6)
    r_pred, sigma2_pred = ag.one_step_prediction(X, estimates, p, q)
    r_pred = r_pred * 0.01
    sigma2_pred = sigma2_pred * 0.01**2
    value_at_risk90 = ag.VaR('norm', r_pred, np.sqrt(sigma2_pred), 1-0.90)
    value_at_risk95 = ag.VaR('norm', r_pred, np.sqrt(sigma2_pred), 1-0.95)
    value_at_risk99 = ag.VaR('norm', r_pred, np.sqrt(sigma2_pred), 1-0.99)
    with open('/home/howardwong/Desktop/Research/ARMA-GARCH-Models/data/aapl_data/{}.txt'.format(prediction_date), 'w') as f:
        f.write(prediction_date + ',' + str(p) + ',' + str(q) + ',' + str(log_returns.values[t2+1]) + ',' + str(value_at_risk90) + ',' + str(value_at_risk95) + ',' + str(value_at_risk99))
        f.close()


if __name__=='__main__':
    pool = mp.Pool(mp.cpu_count())
    pool.map(one_step_var, intervals)