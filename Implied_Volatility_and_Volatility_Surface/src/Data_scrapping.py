import pandas as pd
from datetime import datetime, timedelta
from yahooquery import Ticker
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
from time import mktime

def get_market_data_about_option_chain_for_computing_implied_volatility(tickers_set):
    # Function to retrieve the option chain listed for specified tickers 
    # Making new dir in your folder and retriving data about option chain
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
        
#----------------------------------------------------------------------------
'''
Function to retrieve the option chain listed for specified crypto currency from Derebit exchange 
Making file with option Data in your current folder 
Currency: ETH, BTC,
'''

def get_data_about_crypto_options(currency):
    #scrapping data about curenct active option instruments
    url = 'https://history.deribit.com/api/v2/public/get_instruments?currency=' + currency + '&kind=option&expired=false'
    api_response = (requests.get(url)).json()
    data_options = pd.DataFrame()
    #scrapping data about curenct currency price in USD
    url_spot = 'https://deribit.com/api/v2/public/get_index?currency=' + currency
    spot_price = (requests.get(url_spot)).json()['result']['edp']

    #scrapping data about crypto options
    for k in api_response['result']:
        strike = k['strike']
        option_type = k['option_type']
        time1 = datetime.fromtimestamp(k['expiration_timestamp'] / 1000)
        expiration =  str(time1.year) + '-' + str(time1.month) + '-' + str(time1.day)
        time1 = datetime.fromtimestamp(k['creation_timestamp'] / 1000)
        last_trade_day =  str(time1.year) + '-' + str(time1.month) + '-' + str(time1.day) + ' ' + str(time1.hour) + ':' + str(time1.minute) + ':' + str(time1.second)

        instrument_name = k['instrument_name']
        # scrapping data about given order book
        instuments_information_url = 'https://deribit.com/api/v2/public/get_order_book?instrument_name=' + instrument_name
        instuments_information = (requests.get(instuments_information_url)).json()
        bid_price = instuments_information['result']['best_bid_price']
        ask_price = instuments_information['result']['best_ask_price']
        mid_price = (ask_price + bid_price) / 2
        volume = instuments_information['result']['stats']['volume']
        open_interest = instuments_information['result']['open_interest']
        DerebitIV = (instuments_information['result']['bid_iv'] + instuments_information['result']['ask_iv']) / 2
        data_options = pd.concat([data_options, pd.DataFrame([instrument_name, last_trade_day, expiration, strike, option_type, spot_price, bid_price,ask_price, mid_price, volume, open_interest, DerebitIV]).T])
    
    #making necessary adjustment
    data_options.columns = ['instrumentName', 'lastTradeDate', 'expiryDate', 'strike','optionType','last close', 'bid', 'ask', 'mid', 'volume', 'openInterest', 'DerebitIV']
    data_options['yFinance_dividend_yield'] = 0
    data_options['optionType'].loc[data_options['optionType'] == 'call'] = 'calls'
    data_options['optionType'].loc[data_options['optionType'] == 'put'] = 'puts'
    data_options['ticker'] = currency
    data_options['bid'] = data_options['bid'] * data_options['last close'].unique()[0]
    data_options['ask'] = data_options['ask'] * data_options['last close'].unique()[0]
    data_options['mid'] = data_options['mid'] * data_options['last close'].unique()[0]
    data_options = data_options.loc[data_options['bid'] != 0]
    data_options = data_options.loc[data_options['ask'] != 0]

    data_options.to_csv('DATA//Crypto_currencies//' + currency + '//' + str(datetime.now().date()) + '-' + str(datetime.now().hour) + '-' + str(datetime.now().minute) + '.csv', index=False)

def get_funding_rate(currency):
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    start_timestamp = int(mktime(datetime.strptime(str(yesterday), "%Y-%m-%d").timetuple())) * 1000
    end_timestamp = int(mktime(datetime.strptime(str(today), "%Y-%m-%d").timetuple())) * 1000
    url = 'https://deribit.com/api/v2/public/get_funding_rate_history?instrument_name=' + currency + '-PERPETUAL&start_timestamp=' + str(start_timestamp) +'&end_timestamp=' + str(end_timestamp)
    api_response = requests.get(url).json()

    funding_rate = pd.DataFrame()
    for k in api_response['result']:
        time = datetime.fromtimestamp(k['timestamp'] / 1000)
        Date = str(time.year) + '-' + str(time.month) + '-' + str(time.day)
        rate = k['interest_1h']
    funding_rate = pd.concat([funding_rate, pd.DataFrame([Date,rate]).T])
    funding_rate.columns = ['Date', 'rate']
    funding_rate = funding_rate.set_index('Date')
    rate = funding_rate.mean() * 365
    
    return rate[0]