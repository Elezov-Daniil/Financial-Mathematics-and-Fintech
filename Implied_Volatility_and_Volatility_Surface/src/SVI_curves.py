import matplotlib.pyplot as plt
import numpy as np
from SVI_calibrator import SVI
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
#-------------------------------------------------------------------------------------
'''
Function obtains data about implied volatility surface and calibrates parameters for each curves for SVI model in raw parametrization
Function returns implied volatility computed via SVI model for, calibrated parameters, max relative error between given implied volatility and computed implied volatility; saves data about previous calibration.
'''

def computing_SVI_IV(data_dict, ticker, low_limit = 0.1, high_limit = 2.5, N = 1000, extrapolation=True):
    # download initial parameters from previous calibration
    initial_parameters = pd.read_csv('DATA\\Initial_parametres_for_SVI_calibrator\\List_initial_parameters_for_SVI.csv', index_col='expiry_date')
    initial_parameters_last = initial_parameters.loc[initial_parameters['date'] == initial_parameters['date'].unique()[-1]] #take last calibration
    current_date = str(data_dict['data_date'].year) + '-' + str(data_dict['data_date'].month) +'-' + str(data_dict['data_date'].day)
    initial_parameters_new = pd.DataFrame(columns = initial_parameters.columns, index=['a', 'b', 'rho', 'm', 'sigma'])
    
    
    for k in data_dict['implied_volatility_surface']:
        expiry_date = k['expiry_date']
        w_SVI = 'Fall'
        counter_Fall = 0
        
        if expiry_date in initial_parameters.columns: # find last parameters for given expiration
            initial_param = SVI(initial_parameters_last.loc['a', expiry_date], initial_parameters_last.loc['b', expiry_date], initial_parameters_last.loc['rho', expiry_date], initial_parameters_last.loc['m', expiry_date], initial_parameters_last.loc['sigma', expiry_date])
        else:
            initial_param = []
        while w_SVI == 'Fall': #sometimes functions from optimize library could fall and we need to repeat calibration
            counter_Fall += 1
            w_SVI, max_relative_error, set_param_raw, w_SVI_total_variance = compute_SVI(k, ticker, low_limit, high_limit, N, initial_param, extrapolation) #compute implied volatility from SVI
        counter_Fall = counter_Fall - len(data_dict['implied_volatility_surface']) #count number of fall
        k['SVI_implied_volatilities'] = w_SVI
        k['max_relative_error'] = max_relative_error
        k['set_param_raw'] = set_param_raw
        k['w_SVI_total_variance'] = w_SVI_total_variance
        
        # save new calibrated parameters
        initial_parameters_new.loc['a', expiry_date] = set_param_raw.a
        initial_parameters_new.loc['b', expiry_date] = set_param_raw.b
        initial_parameters_new.loc['rho', expiry_date] = set_param_raw.rho
        initial_parameters_new.loc['m', expiry_date] = set_param_raw.m
        initial_parameters_new.loc['sigma', expiry_date] = set_param_raw.sigma
        
        #compute max relative error between implied volatility from SVI and bid ask spread given implied volatility
        for j in range(0,len(max_relative_error)):
            if (k['SVI_implied_volatilities'][j] >= k['bid_implied_volatilities'][j]) & (k['SVI_implied_volatilities'][j] <= k['ask_implied_volatilities'][j]):
                k['max_relative_error'][j] = 0
            else:
                if k['SVI_implied_volatilities'][j] > k['ask_implied_volatilities'][j]:
                    k['max_relative_error'][j] = k['SVI_implied_volatilities'][j] - k['ask_implied_volatilities'][j]
                else:
                    k['max_relative_error'][j] = k['bid_implied_volatilities'][j] - k['SVI_implied_volatilities'][j]
    
    # keep new calibrated parameters
    initial_parameters_new['date'] = current_date
    if not current_date in initial_parameters['date'].unique():
        initial_parameters = pd.concat([initial_parameters, initial_parameters_new])
    initial_parameters.to_csv('DATA\\Initial_parametres_for_SVI_calibrator\\List_initial_parameters_for_SVI.csv', index= True, index_label='expiry_date')
        
    return data_dict

#-----------------------------------------------
# compute implied volatility for given parameteres and log-moneyness: x = ln(strike/forward)
def get_w_SVI_raw(set_param_raw, x):
    return set_param_raw.a + set_param_raw.b*(set_param_raw.rho*(x-set_param_raw.m) + np.sqrt((x-set_param_raw.m)**2 + set_param_raw.sigma**2))

#-----------------------------------------------
'''
Function obtains data about implied volatility and calibrates parameters for SVI model
Function returns implied volatility computed via SVI model, calibrated parameters, max relative error between given implied volatility and computed implied volatility.
'''
def compute_SVI(k, ticker, low_limit, high_limit, N, initial_param, extrapolation=True):
    x_array = np.log(np.array(k['strikes'])/k['reference_forward']) # log-moneyness
    w_array = np.power(k['mid_implied_volatilities'],2) * k['expiry_date_in_act365_year_fraction'] #total varience
    #choose initial parameters
    a = 1 / 2 * np.min(w_array)
    if initial_param:
        set_param = initial_param
    else:
        set_param = SVI(a, 0.1, -0.5, 0.1, 0.1)
    
    #calibrate SVI parameters, sometimes functions from optimize library could fall and we need to repeat calibration
    try:
        set_param_raw = set_param.calibrate(x_array,w_array) #algoritm based on code of Zhitluhin M.V.
    except Exception:
        return "Fall", None, None, None

    # compute relative error 
    w_SVI_relative_error = np.linspace(0,0,len(x_array))
    for ind,i in enumerate(x_array):
        w_SVI_relative_error[ind] = get_w_SVI_raw(set_param_raw, i) # compute total varience for given parameters
    w_SVI_relative_error = np.sqrt(w_SVI_relative_error / k['expiry_date_in_act365_year_fraction']) #convert to implied volatility
    max_error = w_SVI_relative_error - np.array(k['mid_implied_volatilities'])
    max_relative_error = max_error / np.array(k['mid_implied_volatilities'])
    
    # give value for log-moneyness
    if extrapolation:
        x_grid = np.log(np.linspace(low_limit, high_limit, N))
        w_SVI_raw = np.linspace(0,0,N)
    else:
        x_grid = x_array
        w_SVI_raw = np.linspace(0,0,len(x_array))
    
    # compute implied volatility
    for ind, i in enumerate(x_grid):
        w_SVI_raw[ind] = get_w_SVI_raw(set_param_raw, i) # compute total varience for given parameters
    w_SVI_total_variance = w_SVI_raw
    w_SVI_raw = np.sqrt(w_SVI_total_variance / k['expiry_date_in_act365_year_fraction']) #convert to implied volatility

    return w_SVI_raw, max_relative_error, set_param_raw, w_SVI_total_variance
