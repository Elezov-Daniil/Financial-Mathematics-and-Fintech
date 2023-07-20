import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from SVI_curves import get_w_SVI_raw

#----------------------------------------------------------------------------
'''
Function computes option prices for given implied volatility
Function returns dataframe with option prices
'''
def get_data_about_option(data_dict, options_data, optionTypes=None): 
    low_limit = 0.1
    high_limit = 2.5
    N = 1000
    value_options = pd.DataFrame(columns=['x_grid', 'IV', 'value_option', 'expiry_date','expiry_date_in_act365_year_fraction','optionType', 'strike', 'bid', 'ask'])
#     options_data = pd.read_csv(filename)
    
    for k in data_dict["implied_volatility_surface"]:
        Option_value = pd.DataFrame()
        set_param_raw = k['set_param_raw']
        T = k['expiry_date_in_act365_year_fraction']
        F = k['reference_forward']
        expiry_date = k['expiry_date']
        discount_factor = k['reference_discount_factor']
        
        #if necessary keep only out of the market options
        if optionTypes == 'OTM':
            options_data_expiry_date = options_data.loc[options_data["expiryDate"]==expiry_date]
            options_data_expiry_date = options_data_expiry_date.loc[((options_data_expiry_date["strike"] >= F) & (options_data_expiry_date["optionType"] == "calls")) | ((options_data_expiry_date["strike"] < F) & (options_data_expiry_date["optionType"] == 'puts'))]
            options_data_expiry_date = options_data_expiry_date.sort_values(by=['strike'])
        else: 
            options_data_expiry_date = options_data
        
        stirkes = (options_data_expiry_date.loc[options_data_expiry_date["expiryDate"]==expiry_date])['strike'] # take all strikes for this expirations
        optinType = (options_data_expiry_date.loc[options_data_expiry_date["expiryDate"]==expiry_date])['optionType'] # take corresponding calls and puts
        
        for ind,j in enumerate(stirkes.unique()):
            x = np.log(j/F) # compute log moneyness
            IV = np.sqrt(get_w_SVI_raw(set_param_raw, x) / T) # compute implied volatility from SVI model
            bid_ask = options_data_expiry_date.loc[(options_data_expiry_date["expiryDate"]==expiry_date) & (options_data_expiry_date["strike"]==j)] # take bid ask spread correspond given strike
            optionType = (options_data_expiry_date.loc[(options_data_expiry_date["expiryDate"]==expiry_date) & (options_data_expiry_date["strike"]==j)])['optionType'] # take option type (call and put) correspond given strike
            
            #compute option prices via Black 76 model
            if len(optionType) == 2: # we have put and call correspond to this strike
                Calls = black_price_formula(F, j, T, IV, 'calls') 
                Puts = black_price_formula(F, j, T, IV, 'puts')
                bid_ask_calls = bid_ask.loc[bid_ask['optionType'] == 'calls']
                bid_ask_puts = bid_ask.loc[bid_ask['optionType'] == 'puts']
                Option_value = pd.concat([Option_value, pd.DataFrame([np.exp(x), IV, Calls * discount_factor , expiry_date, T, 'calls', j, bid_ask_calls['bid'].item(), bid_ask_calls['ask'].item()]).T])
                Option_value = pd.concat([Option_value, pd.DataFrame([np.exp(x), IV, Puts * discount_factor, expiry_date, T, 'puts', j,bid_ask_puts['bid'].item(), bid_ask_puts['ask'].item()]).T])
            else:
                if (optionType.iloc[0] == 'calls'): # we have only call correspond to this strike
                    Calls = black_price_formula(F, j, T, IV, 'calls')
                    Option_value = pd.concat([Option_value, pd.DataFrame([np.exp(x), IV, Calls * discount_factor, expiry_date, T,'calls', j,bid_ask['bid'].item(), bid_ask['ask'].item()]).T])
                else:  # we have only put correspond to this strike
                    Puts = black_price_formula(F, j, T, IV, 'puts')
                    Option_value = pd.concat([Option_value, pd.DataFrame([np.exp(x), IV, Puts * discount_factor, expiry_date, T,'puts', j,bid_ask['bid'].item(), bid_ask['ask'].item()]).T])
        
        Option_value.columns = ['x_grid', 'IV', 'value_option', 'expiry_date','expiry_date_in_act365_year_fraction','optionType', 'strike', 'bid', 'ask']

        value_options = pd.concat([value_options, Option_value])
        
        # find such option prices, which are within the initial bid ask spread
        value_options['correct_price'] = value_options.apply(lambda x: 'inside' if ((x.value_option <= x.ask) & (x.value_option >= x.bid)) else ('below' if (x.value_option < x.bid) else 'above'), axis=1) 
            
    return value_options


#----------------------------------------------------------------------------
#Functions compute option value for given parameters 
def black_price_formula(F, K, T, sigma, optionType):
        stddev = sigma*np.sqrt(T)
        d1 = np.log(F/K)/stddev + 0.5*stddev
        d2 = d1 - stddev
        eps = 1.0 if optionType == 'calls' else -1.0
        return eps *(F * norm.cdf(eps*d1) - K * norm.cdf(eps*d2))