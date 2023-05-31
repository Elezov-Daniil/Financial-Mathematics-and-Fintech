import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#-------------------------------------------------------------------
'''
Function checks 3 types of arbitrage: calendar, butterflies and asymptotics behaviour for large and small strikes
Function prints if some kind of arbitrage is exist
'''
def check_static_arbitrage(value_options, data_dict):
    calendar_arbitrage = check_calendar_arbitrage(value_options)
    butterfly_arbitrage = check_butterflies_arbitrage_with_formula(data_dict)
    limit_price = check_limit_price(data_dict)
    print()
    if len(calendar_arbitrage) + len(butterfly_arbitrage) + len(limit_price) == 0:
        print('static_arbitrage is absence')
    else:
        
        if len(calendar_arbitrage) != 0:
            print('calandar arbitrage exists')
        if len(butterfly_arbitrage) != 0:
            print('butterfly arbitrage exists')
        if len(limit_price) != 0:
            print('limit price arbitrage exists')
    

#--------------------------------------------------------
'''
Function checks calender arbitrage, based on article "Arbitrage-free SVI volatility surfaces" by
Jim Gatheral and Antoine Jacquiery. We have to compare value option in term of bid for previous expiration with value option in term of ask for further expiration.
Function returns expirations, when calendar arbitrage exists.
'''

def check_calendar_arbitrage(value_options):
    value_option_check = value_options
    strike = value_option_check['strike'].unique()
    calendar_arbitrage = pd.DataFrame()
    for k in strike:
        Calls = value_option_check.loc[(value_option_check['strike'] == k) & (value_option_check['optionType'] == 'calls')] # take calls for specified strike
        Puts = value_option_check.loc[(value_option_check['strike'] == k) & (value_option_check['optionType'] == 'puts')] # take puts for specified strike
        Calls = Calls.sort_values(by='expiry_date')
        Puts = Puts.sort_values(by='expiry_date')
             # compare value options between different expiration and find those, which value less than in previous expiration in term of bid ask spread
        c_0 = 0
        spread_size_last = 0
        for i in range(0, len(Calls)):
            spread_size = (Calls['ask'].iloc[i] - Calls['bid'].iloc[i]) / 2
            c_1 = Calls['value_option'].iloc[i] + spread_size # find value option in term of ask
            if c_0 < c_1:
                c_0 = c_1 - spread_size * 2 #find value option in term of bid
                spread_size_last = spread_size
            else:
                calendar_arbitrage = pd.concat([calendar_arbitrage, Calls.loc[(Calls['value_option'] == c_0 + spread_size_last) | (Calls['value_option'] == c_1 - spread_size)]])
                c_0 = c_1 - spread_size * 2
                spread_size_last = spread_size

        spread_size_last = 0
        p_0 = 0
        for i in range(0, len(Puts)):
            spread_size = (Puts['ask'].iloc[i] - Puts['bid'].iloc[i]) / 2
            p_1 = Puts['value_option'].iloc[i] + spread_size # find value option in term of ask
            if p_0 < p_1:
                p_0 = p_1 - spread_size * 2 #find value option in term of bid
                spread_size_last = spread_size
            else:
                calendar_arbitrage = pd.concat([calendar_arbitrage, Puts.loc[(Puts['value_option'] == p_0 + spread_size_last) | (Puts['value_option'] == p_1 - spread_size)]])
                p_0 = p_1 - spread_size * 2
                spread_size_last = spread_size

    return calendar_arbitrage
#-------------------------------------------------------------
'''
Function checks butterflies arbitrage, based on article "Robust Calibration For SVI Model Arbitrage Free" by Tahar Ferhati.
    (a - m*b*(rho + 1))*(4 - a + m*b*(rho + 1)) / b^2*(rho + 1)^2 > 1
    (a - m*b*(rho - 1))*(4 - a + m*b*(rho - 1)) / b^2*(rho - 1)^2 > 1
    0 < b^2*(rho - 1)^2 < 4
    0 < b^2*(rho + 1)^2 < 4
Function returns expirations, when butterflies arbitrage exists.
'''
def check_butterflies_arbitrage_with_formula(data_dict):    
    butterflies_arbitrage = []
    for k in data_dict["implied_volatility_surface"]:
        set_param = k['set_param_raw']
        if not (((set_param.a - set_param.m * set_param.b * (set_param.rho + 1)) * (4 - set_param.a + set_param.m * set_param.b * (set_param.rho + 1))) / (set_param.b ** 2 * (set_param.rho + 1) ** 2) > 1) & (((set_param.a - set_param.m * set_param.b * (set_param.rho - 1)) * (4 - set_param.a + set_param.m * set_param.b * (set_param.rho - 1))) / (set_param.b ** 2 * (set_param.rho - 1) ** 2) > 1) & (set_param.b ** 2 * (set_param.rho - 1) ** 2 < 4) & (set_param.b ** 2 * (set_param.rho - 1) ** 2 > 0) & (set_param.b ** 2 * (set_param.rho + 1) ** 2 < 4) & (set_param.b ** 2 * (set_param.rho + 1) ** 2 > 0):
            butterflies_arbitrage.append(k['expiry_date'])
            
    return butterflies_arbitrage
#-----------------------------------------------------------------------
'''
Function checks the asymptotics behaviour for large and small strikes (1 + |rho| * b ⩽ 2).
Function returns expirations, when arbitrage exists.
'''

def check_limit_price(data_dict):
    limit_price = []
    for k in data_dict["implied_volatility_surface"]:
        set_param = k['set_param_raw']
        if not (set_param.b * (1 + set_param.rho) < 2):
            limit_price.append(k['expiry_date'])
    return limit_price

#-----------------------------------------------------------------------
'''
Function computes g-function from article "Arbitrage-free SVI volatility surfaces" by
Jim Gatheral and Antoine Jacquiery for given expiration and calibrated parameters.
g(x) := (1 - xw'(x)/2w(x))^2 - w'(x)^2/4 * (1/w(x) + 1/4) + w''(x)/2

Function returns g-function value and plots graph.
'''

def get_g_function_for_butterfly_arbitrage(data_dict, expiry_date, graph=None, N=1000, right_limit = 3):
    set_param = data_dict['implied_volatility_surface'][expiry_date]['set_param_raw'] # calibrated parameters
    expiry_date = data_dict['implied_volatility_surface'][expiry_date]['expiry_date'] # expiration date
    k = np.log(np.linspace(0.1, right_limit,N)) # log - moneyness
    w_k = np.linspace(0,0,N) # total varience
    for ind,i in enumerate(k):
        w_k[ind] = set_param.a + set_param.b * (set_param.rho * (i - set_param.m) + np.sqrt((i - set_param.m) ** 2 + set_param.sigma ** 2))
    w_k_first = np.linspace(0,0,len(w_k)) # first derivatives by the strike 
    w_k_second = np.linspace(0,0,len(w_k))# second derivatives by the strike 
    g_k = np.linspace(0,0,len(w_k))
    # compute necessary data
    for i in range(0, len(w_k) - 2):
        w_k_first[i+1] = (w_k[i+2] - w_k[i]) / (k[i+2] - k[i])
        w_k_second[i+1] = (w_k[i+2] - 2 * w_k[i+1] + w_k[i]) / (k[i+2] - k[i]) ** 2
    w_k_first[0] = (w_k[1] - w_k[0]) / (k[1] - k[0])
    w_k_second[0] = (w_k[1] - w_k[0]) / (k[1] - k[0])
    w_k_first[len(w_k)-1] = (w_k[len(w_k)-1] - w_k[len(w_k)-2]) / (k[len(w_k)-1] - k[len(w_k)-2])
    w_k_second[len(w_k)-1] = (w_k[len(w_k)-1] - w_k[len(w_k)-2]) / (k[len(w_k)-1] - k[len(w_k)-2])
    
    # compute g-function
    for i in range(0, len(w_k)):
        g_k[i] = (1 - k[i] * w_k_first[i] / 2 / w_k[i]) ** 2 - w_k_first[i] ** 2 / 4 * (1 / w_k[i] + 1 / 4) + w_k_second[i] / 2
        g_k[0] = g_k[1]
        g_k[-1] = g_k[-2]
    
    if graph:
        fig = plt.figure(figsize=(13, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.plot(k,w_k)
        ax2.plot(k,g_k)

        ax1.set_title(expiry_date + ' implied volatility from SVI')
        ax1.set_xlabel(r'log moneyness)')
        ax1.set_ylabel(r'Implied Volatility')
        ax1.grid()
        ax2.set_title(expiry_date + ' g-function from SVI')
        ax2.set_xlabel(r'log moneyness')
        ax2.set_ylabel(r'function g')
        ax2.grid()
        plt.show()
    return g_k

