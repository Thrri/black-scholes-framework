import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt

def call_option(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-div*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def put_option(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-div*T)*norm.cdf(-d1)

def call_delta(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return np.exp(-div*T)*norm.cdf(d1)

def put_delta(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return -np.exp(-div*T)*norm.cdf(-d1)
    
def option_gamma(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return np.exp(-div*T)*norm.pdf(d1)/(S*sigma*np.sqrt(T))

def option_vega(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return S*np.exp(-div*T)*norm.pdf(d1)*np.sqrt(T)

def call_theta(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return -np.exp(-div*T)*S*norm.pdf(d1)*sigma/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2)+div*S*np.exp(-div*T)*norm.cdf(d1)

def put_theta(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return -np.exp(-div*T)*S*norm.pdf(d1)*sigma/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2)-div*S*np.exp(-div*T)*norm.cdf(-d1)

def call_rho(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*T*np.exp(-r*T)*norm.cdf(d2)

def put_rho(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return -K*T*np.exp(-r*T)*norm.cdf(-d2)

def call_epsilon(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return -S*T*np.exp(-div*T)*norm.cdf(d1)

def put_epsilon(S, K, sigma, T, r, div):
    d1 = (np.log(S/K) + (r - div + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return S*T*np.exp(-div*T)*norm.cdf(-d1)

def call_omega(S, K, sigma, T, r, div):
    return call_delta(S, K, sigma, T, r, div) * S / call_option(S, K, sigma, T, r, div)

def put_omega(S, K, sigma, T, r, div):
    return put_delta(S, K, sigma, T, r, div) * S / put_option(S, K, sigma, T, r, div)

def call_standard_deviation(S, K, sigma, T, r, div):
    return np.abs(call_omega(S, K, sigma, T, r, div))*sigma

def put_standard_deviation(S, K, sigma, T, r, div):
    return np.abs(put_omega(S, K, sigma, T, r, div)) * sigma


def options_info(S, K, sigma, T, r, div, plotting):

    call = call_option(S, K, sigma, T, r, div)
    put = put_option(S, K, sigma, T, r, div)

    delta_c = call_delta(S, K, sigma, T, r, div)
    delta_p = put_delta(S, K, sigma, T, r, div)

    gamma_c = option_gamma(S, K, sigma, T, r, div)
    gamma_p = option_gamma(S, K, sigma, T, r, div)

    vega_c = option_vega(S, K, sigma, T, r, div)
    vega_p = option_vega(S, K, sigma, T, r, div)

    theta_c = call_theta(S, K, sigma, T, r, div)
    theta_p = put_theta(S, K, sigma, T, r, div)

    rho_c = call_rho(S, K, sigma, T, r, div)
    rho_p = put_rho(S, K, sigma, T, r, div)

    epsilon_c = call_epsilon(S, K, sigma, T, r, div)
    epsilon_p = put_epsilon(S, K, sigma, T, r, div)

    omega_c = call_omega(S, K, sigma, T, r, div)
    omega_p = put_omega(S, K, sigma, T, r, div)

    standard_deviation_c = call_standard_deviation(S, K, sigma, T, r, div)
    standard_deviation_p = put_standard_deviation(S, K, sigma, T, r, div)

    variable_names =['Call price', 'Put price', 
                     'Call delta','Put delta',
                     'Call gamma','Put gamma',
                     'Call vega','Put vega',
                     'Call theta','Put theta',
                     'Call rho','Put rho',
                     'Call epsilon','Put epsilon',
                     'Call elasticity','Put elasticity',
                     'Call standard deviation', 'Put Standard deviation']
    
    option_data = [call, put, 
                   delta_c, delta_p, 
                   gamma_c, gamma_p, 
                   vega_c, vega_p, 
                   theta_c, theta_p, 
                   rho_c, rho_p, 
                   epsilon_c, epsilon_p, 
                   omega_c, omega_p, 
                   standard_deviation_c, standard_deviation_p]
    
    option_info = dict(zip(variable_names, option_data))

    counter = 0
    print()
    for key, value in option_info.items():
        print(f'{key}: {value}')
        counter += 1
        if counter % 2 == 0:
            print()
    

    if plotting:
        St_values = np.linspace(0.01, 2*S, 2*S)
        call_prices = []
        call_deltas = []
        call_gammas = []

        put_prices = []
        put_deltas = []
        put_gammas = []

        for i in range(len(St_values)):
            price_call = call_option(St_values[i], K, sigma, T, r, div)
            delta_call = call_delta(St_values[i], K, sigma, T, r, div)
            gamma_call = option_gamma(St_values[i], K, sigma, T, r, div)

            price_put = put_option(St_values[i], K, sigma, T, r, div)
            delta_put = put_delta(St_values[i], K, sigma, T, r, div)
            gamma_put = option_gamma(St_values[i], K, sigma, T, r, div)

            call_prices.append(price_call)
            call_deltas.append(delta_call)
            call_gammas.append(gamma_call)

            put_prices.append(price_put)
            put_deltas.append(delta_put)
            put_gammas.append(gamma_put)


        plt.figure(figsize=(16, 9))

        plt.subplot(2, 3, 1)
        plt.plot(St_values, call_prices, color='green')
        plt.title('Call price vs. St')

        plt.subplot(2, 3, 2)
        plt.plot(St_values, call_deltas, color='green')
        plt.title('Call delta vs. St')
        
        plt.subplot(2, 3, 3)
        plt.plot(St_values, call_gammas, color='green')
        plt.title('Call gamma vs. St')
        
        plt.subplot(2, 3, 4) 
        plt.plot(St_values, put_prices, color='red')
        plt.title('Put price vs. St')
        
        plt.subplot(2, 3, 5)
        plt.plot(St_values, put_deltas, color='red')
        plt.title('Put delta vs. St')
        
        plt.subplot(2, 3, 6)
        plt.plot(St_values, put_gammas, color='red')
        plt.title('Put gamma vs. St')
        
        plt.tight_layout()
        plt.show()


options_info(S=100, K=96, sigma=0.5, T=3, r=0.05, div=0.02, plotting=True)

