import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def get_Implied_volatility(options_data, SOFR_curve, expiry_date_list=None):
    data_date = datetime.strptime(options_data["lastTradeDate"].iloc[0], "%Y-%m-%d %H:%M:%S")
    
    Implied_Volatility={}
    Implied_Volatility["underlying_ticker"]= options_data["ticker"].iloc[0]
    Implied_Volatility["data_date"] = data_date
    Implied_Volatility["reference_spot"] = reference_spot = float(options_data["last close"].iloc[0])
    Implied_Volatility_surface =list()
    
    if not expiry_date_list:
        expiry_date_list=options_data["expiryDate"].unique().tolist()
    
    for expiry_date in expiry_date_list:
        options_data_expiry_date = options_data.loc[options_data["expiryDate"]==expiry_date]
        SOFR_discount_rate_dayfraction = SOFR_curve['date settlement'].map(lambda x : ((x- data_date).days)/360)
        SOFR_discount_rate = SOFR_curve['close'] / 100
        T_actual_360 = ((datetime.strptime(expiry_date, "%Y-%m-%d") - data_date).days)/360
        T_actual_365 = ((datetime.strptime(expiry_date, "%Y-%m-%d") - data_date).days)/365
        discount_factor = np.exp(np.interp(T_actual_360, SOFR_discount_rate_dayfraction, SOFR_discount_rate) * T_actual_360)
        reference_forward = get_reference_forward(options_data_expiry_date, expiry_date, SOFR_curve, discount_factor, T_actual_365)
        
        options_data_expiry_date = computing_implied_volatility(options_data_expiry_date, reference_forward, discount_factor, T_actual_365, strike_upper_limit = 2.5, strike_lower_limit = 0.4, delta_limit = 0.001, IV_bis_ask_spread_quantile = 0.95)
        
        Implied_Volatility_for_specific_expiry={}
        Implied_Volatility_for_specific_expiry["expiry_date"] = expiry_date
        Implied_Volatility_for_specific_expiry["number_of_days_from_value_date"] = T_actual_365 * 365
        Implied_Volatility_for_specific_expiry["expiry_date_in_act365_year_fraction"] = T_actual_365
        Implied_Volatility_for_specific_expiry["reference_forward"] = reference_forward
        Implied_Volatility_for_specific_expiry["reference_discount_factor"] = discount_factor
        Implied_Volatility_for_specific_expiry["strikes"] = options_data_expiry_date["strike"].tolist()
        Implied_Volatility_for_specific_expiry["mid_implied_volatilities"] = options_data_expiry_date["mid_IV"].tolist()
        Implied_Volatility_for_specific_expiry["bid_implied_volatilities"] = options_data_expiry_date["bid_IV"].tolist()
        Implied_Volatility_for_specific_expiry["ask_implied_volatilities"] = options_data_expiry_date["ask_IV"].tolist()

        Implied_Volatility_surface.append(Implied_Volatility_for_specific_expiry)
    Implied_Volatility["implied_volatility_surface"] = Implied_Volatility_surface
        
    return Implied_Volatility
        

def computing_implied_volatility(options_data_expiry_date, reference_forward, discount_factor, T_actual_365, strike_upper_limit = 2.5, strike_lower_limit = 0.4, delta_limit = 0.001, IV_bis_ask_spread_quantile = 0.95):
    options_data_expiry_date = options_data_expiry_date.loc[((options_data_expiry_date["strike"] >= reference_forward) & (options_data_expiry_date["optionType"] == "calls")) | ((options_data_expiry_date["strike"] < reference_forward) & (options_data_expiry_date["optionType"] == 'puts'))]
    options_data_expiry_date = options_data_expiry_date.sort_values(by=['strike'])

        # filter to bondary condition with moneyness
    options_data_expiry_date.drop(options_data_expiry_date[ (options_data_expiry_date.strike<strike_lower_limit*reference_forward) | (options_data_expiry_date.strike>strike_upper_limit*reference_forward)].index, inplace=True)
    
        #compute IV for bid, ask, mid
    options_data_expiry_date["mid_IV"] = options_data_expiry_date.apply(lambda x: black_implied_vol(x.mid * discount_factor, reference_forward, x.strike, T_actual_365, x.optionType), axis=1)
    options_data_expiry_date["ask_IV"] = options_data_expiry_date.apply(lambda x: black_implied_vol(x.ask * discount_factor, reference_forward, x.strike, T_actual_365, x.optionType), axis=1)
    options_data_expiry_date["bid_IV"] = options_data_expiry_date.apply(lambda x: black_implied_vol(x.bid * discount_factor, reference_forward, x.strike, T_actual_365, x.optionType), axis=1)
    options_data_expiry_date["IV_bid_ask_spread"] = options_data_expiry_date["ask_IV"] - options_data_expiry_date["bid_IV"]
    options_data_expiry_date.drop(options_data_expiry_date.index[ (options_data_expiry_date["bid_IV"]<0.0) | (options_data_expiry_date["ask_IV"]<0.0)], inplace=True)
        
         #Filter data based on delta criteria
    for strike in options_data_expiry_date["strike"]:
        if strike > reference_forward:
            optionType ='calls' 
        else: 
            optionType='puts'
        IV = options_data_expiry_date.loc[(options_data_expiry_date["strike"]==strike) & (options_data_expiry_date["optionType"]== optionType)]["mid_IV"].item()
        delta = black_delta_formula(reference_forward, strike, T_actual_365, IV, optionType)
        if abs(delta)<=delta_limit:
            options_data_expiry_date.drop(options_data_expiry_date.index[options_data_expiry_date["strike"]==strike], inplace=True)

    #cleaning of remaining data based on the size of the bid-ask spread
    bid_ask_spread_limit =  options_data_expiry_date["IV_bid_ask_spread"].quantile(IV_bis_ask_spread_quantile)
    if len(options_data_expiry_date)>=10:
        options_data_expiry_date.drop(options_data_expiry_date[options_data_expiry_date["IV_bid_ask_spread"] > bid_ask_spread_limit].index, inplace=True)
    
    return options_data_expiry_date
    
def get_reference_forward(options_data_expiry_date, expiry_date, SOFR_curve, discount_factor, T_actual_365):
    spot_value = options_data_expiry_date["last close"].iloc[0]
    dividend_yield = options_data_expiry_date["yFinance_dividend_yield"].iloc[0]
    reference_forward = spot_value * discount_factor * np.exp(- dividend_yield * (T_actual_365)) #no repo rate
    
    optionsTduplicateStrikes = options_data_expiry_date[options_data_expiry_date.duplicated(["strike"], keep=False)]
    optionsTduplicateStrikes = optionsTduplicateStrikes.sort_values(by=['strike'], ascending = [False])

    if (len(optionsTduplicateStrikes)==0): #degenerated case
        return reference_forward

    elif len(optionsTduplicateStrikes) == 2: #there are two lines, hence only one strike
        strike = optionsTduplicateStrikes.iloc[0]["strike"].item()
        mid_call_price = optionsTduplicateStrikes.loc[(optionsTduplicateStrikes["strike"]==strike) & (optionsTduplicateStrikes["optionType"]=='calls')]["mid"].item()
        mid_put_price = optionsTduplicateStrikes.loc[(optionsTduplicateStrikes["strike"]==strike) & (optionsTduplicateStrikes["optionType"]=='puts')]["mid"].item()
        reference_forward = (mid_call_price - mid_put_price) * discount_factor + strike

    else : #regular case, i.e. len(optionsTduplicateStrikes)>=4
        strikes = find_neighbours(reference_forward, optionsTduplicateStrikes, "strike")
        #computation of the adjusted fwd from the strike just below the reference fwd
        strike_below_fwd = optionsTduplicateStrikes.loc[strikes[0]]["strike"]
        mid_call_price = optionsTduplicateStrikes.loc[(optionsTduplicateStrikes["strike"]==strike_below_fwd) & (optionsTduplicateStrikes["optionType"]=='calls')]["mid"].item()
        mid_put_price = optionsTduplicateStrikes.loc[(optionsTduplicateStrikes["strike"]==strike_below_fwd) & (optionsTduplicateStrikes["optionType"]=='puts')]["mid"].item()
        fwd_from_strike_below_fwd = (mid_call_price-mid_put_price) * discount_factor + strike_below_fwd
        #computation of the adjusted fwd from the strike just above the reference fwd
        strike_above_fwd = optionsTduplicateStrikes.loc[strikes[1]]["strike"]
        mid_call_price = optionsTduplicateStrikes.loc[(optionsTduplicateStrikes["strike"]==strike_above_fwd) & (optionsTduplicateStrikes["optionType"]=='calls')]["mid"].item()
        mid_put_price = optionsTduplicateStrikes.loc[(optionsTduplicateStrikes["strike"]==strike_above_fwd) & (optionsTduplicateStrikes["optionType"]=='puts')]["mid"].item()
        fwd_from_strike_above_fwd = (mid_call_price-mid_put_price) * discount_factor + strike_above_fwd
        #average of previous results
        reference_forward = (fwd_from_strike_below_fwd+fwd_from_strike_above_fwd)/2

    return reference_forward

def find_neighbours(value, df, colname):
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        return [lowerneighbour_ind, upperneighbour_ind]

def black_delta_formula(F, K, T, sigma, optionType):
    stddev = sigma*np.sqrt(T)
    d1 = np.log(F/K)/stddev + 0.5*stddev
    eps = 1.0 if optionType == 'calls' else -1.0
    return eps *norm.cdf(eps*d1)
    
    
def black_implied_vol(undiscounted_price, F, K, T, optionType, initial_guess = 0.5):
    target_price = undiscounted_price
    eps = 1.0 if optionType == 'calls' else -1.0
    MAX_ITERATIONS = 200
    PRECISION = min(1.0e-5, target_price/F)
    sigma = initial_guess
    for i in range(0, MAX_ITERATIONS):
        stddev = sigma*np.sqrt(T)
        d1 = np.log(F/K)/stddev + 0.5*stddev
        d2 = d1 - stddev
        price = eps *(F * norm.cdf(eps*d1) - K * norm.cdf(eps*d2))
        vega = F * norm.pdf(d1) * np.sqrt(T)
        diff = target_price - price
        if (vega<=np.finfo(float).eps):
            return initial_guess #likely to be far off the money anyway....
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # Newton–Raphson
    return sigma # if MAX_ITERATIONS, return estimate so far

