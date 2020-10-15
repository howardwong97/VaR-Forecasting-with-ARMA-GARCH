# ARMA-GARCH-Models

This repo documents my general exploration of ARMA-GARCH models, and how I created a Python module for fitting them with Quasi-Maximum Likelihood estimation. I used my findings to run a simple historical backtest to create a one-day-ahead estimate of Value-at-Risk (VaR). 

## 1: Exploring Distribution of Stock Returns 

This notebook goes through some of the key characteristics and statistical distributions of stock returns.

## 2: Theoretical Background of ARMA-GARCH

Here I go through the basic theory of Autoregressive Moving-Average (ARMA) and General Autoregressive Conditional Heteroscedastic (GARCH) processes and why they are relevant.

## 3: ARMA-GARCH MLE Normal Distribution

I demonstrate how Maximum Likelihood Estimation can be applied using the Gaussian likelihood function.

## 4: GARCH with Students-t Distribution

Similar to above, but this time with a different likelihood function.

## 5: Forecasting Value at Risk

Here, I bring all the theory and code from the previous notebooks and perform a backtest of the model to estimate one day 5% VaR values. I also demonstrate how the results show that the model is not as predictive/ accurate as it may seem.
