import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt 

def call_option(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S*np.exp(-div*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

def call_delta(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    delta = np.exp(-div*T)*norm.cdf(d1)
    return delta

def call_gamma(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    gamma = np.exp(-div*T)*norm.pdf(d1)/(S*sigma*np.sqrt(T))
    return gamma

# PARAMS

St = np.linspace(0, 100, 100)
K1 = 50
K2 = 60
r = 0.08
delta = 0
sigma = 0.25
T = 1

call_to_sell = call_option(50, K1, sigma, T, r, delta)
call_to_buy = call_option(50, K2, sigma, T, r, delta)

call_to_sell_delta = call_delta(50, K1, sigma, T, r, delta)
call_to_buy_delta = call_delta(50, K2, sigma, T, r, delta)

call_to_sell_gamma = call_gamma(50, K1, sigma, T, r, delta)  
call_to_buy_gamma = call_gamma(50, K2, sigma, T, r, delta)  

# Set up system of equations 

A = np.array([[call_to_buy_gamma, 0], [call_to_buy_delta, 1]])
B = np.array([call_to_sell_gamma, call_to_sell_delta])
x, y = np.linalg.solve(A, B)

print(x, y)

# CASHFLOWS AT T=0 FOR ALL PORTFOLIOS
original_unhedged = call_to_sell
original_delta_hedged = call_to_sell - call_to_sell_delta*50
original_delta_gamma_hedged = call_to_sell - x*call_to_buy - y*50

# Profit calculations
unhedged_profits = []
delta_hedged_profits = []
delta_gamma_hedged_profits = []

for i in range(len(St)):
    unhedged_profit = original_unhedged - call_option(St[i], K1, sigma, T-1/365, r, delta)*np.exp(r*1/365)
    delta_hedged_profit = original_delta_hedged - (call_option(St[i], K1, sigma, T-1/365, r, delta)-call_to_sell_delta*St[i])*np.exp(r*1/365)
    delta_gamma_hedged_profit = original_delta_gamma_hedged - (call_option(St[i], K1, sigma, T-1/365, r, delta) - x*call_option(St[i], K2, sigma, T-1/365, r, delta) - y*St[i])*np.exp(r*1/365)
    
    unhedged_profits.append(unhedged_profit)
    delta_hedged_profits.append(delta_hedged_profit)
    delta_gamma_hedged_profits.append(delta_gamma_hedged_profit)


plt.figure(figsize=(8, 6))
plt.plot(St, unhedged_profits, label='unhedged')
plt.plot(St, delta_hedged_profits, label='delta-hedged')
plt.plot(St, delta_gamma_hedged_profits, label='delta-gamma-hedged')
plt.ylabel('Profit')
plt.xlabel('1-day stock price ($)')
plt.legend()
plt.show()