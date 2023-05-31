import pandas as pd
from datetime import datetime, timedelta
from yahooquery import Ticker
import matplotlib.pyplot as plt
import numpy as np
import os

def get_market_data_about_option_chain_for_computing_implied_volatility(tickers_set):
    # Frunction to retrieve the option chain listed for specified tickers 
    # making new dir in your folder and retriving data about option chain
    directory = 'DATA//Option_chain//' + str(datetime.now().date()) + '_' + str(datetime.now().hour) + '-' + str(datetime.now().minute)
    os.mkdir(directory)
    for ticker in tickers_set:
        data = Ticker(ticker)
        data_options = data.option_chain
    # making necessary adjustments
        data_options.reset_index(inplace=True)
        data_options.drop(['currency', 'change', 'percentChange', 'contractSize'], axis=1, inplace=True) # Drop unnecessary columns        
        data_options.drop(data_options.loc[data_options['bid'] == 0].index, axis=0, inplace=True) # Drop options without trades
        data_options['last close'] = (Ticker(ticker).history('1d'))['close'][0] # Obtain reference spot
        data_options.drop((data_options.loc[data_options['strike'] > data_options['last close'] * 8]).index, axis=0, inplace=True)
        data_options['mid'] = (data_options['ask'].add(data_options['bid'])) / 2 # Obtain mid price from bia and ask
        data_options = data_options[['contractSymbol', 'symbol', 'lastTradeDate', 'expiration', 'strike', 'lastPrice', 'bid', 'ask', 'mid', 'volume', 'openInterest', 'impliedVolatility', 'last close', 'optionType']]
        data_options = data_options.rename(columns={'symbol' : 'ticker', 'expiration' : 'expiryDate'})
        
    # computing dividend yield
    # dividend hypothesis is that company will pay similar dividends as before.
        frequently_payments_each_year = len(data.dividend_history(start='2022-01-01', end='2023-01-01'))
        average_dividends = (data.dividend_history(start='2022-01-01')).reset_index()
        if average_dividends.empty:
            average_dividends = 0
        else:
            average_dividends = average_dividends['dividends'].mean()
        data_options['yFinance_dividend_yield'] = average_dividends * frequently_payments_each_year /  data_options['last close'].mean()
    
    # keeping collected data in maked dir
        data_folder = directory + '//' + ticker + ".csv"
        data_options.to_csv(data_folder, index= False )
        
