import pandas as pd
from datetime import datetime, timedelta
from yahooquery import Ticker
import matplotlib.pyplot as plt
import numpy as np
import os

def get_market_data_by_option_for_computing_implied_volatility(tickers_set):
    #scrapping and processing market data
    directory = 'DATA-' + str(datetime.now().date()) + '_' + str(datetime.now().hour) + '-' + str(datetime.now().minute)
    os.mkdir(directory)
    for ticker in tickers_set:
        data = Ticker(ticker)
        data_options = data.option_chain
        data_options.reset_index(inplace=True)
        data_options.drop(['currency', 'change', 'percentChange', 'contractSize'], axis=1, inplace=True)
        data_options.drop(data_options.loc[data_options['bid'] == 0].index, axis=0, inplace=True)
        data_options['last close'] = (Ticker(ticker).history('1d'))['close'][0]
        data_options.drop((data_options.loc[data_options['strike'] > data_options['last close'] * 8]).index, axis=0, inplace=True)
        data_options['mid'] = (data_options['ask'].add(data_options['bid'])) / 2
        # data_options = data_options.loc[data_options['inTheMoney'] == False]
        data_options = data_options[['contractSymbol', 'symbol', 'lastTradeDate', 'expiration', 'strike', 'lastPrice', 'bid', 'ask', 'mid', 'volume', 'openInterest', 'impliedVolatility', 'last close', 'optionType']]
        data_options = data_options.rename(columns={'symbol' : 'ticker', 'expiration' : 'expiryDate'})
        
    #computing dividend yield
        dividend_payments_each_year = len(data.dividend_history(start='2022-01-01', end='2023-01-01'))
        average_dividends = (data.dividend_history(start='2022-01-01')).reset_index()
        if average_dividends.empty:
            average_dividends = 0
        else:
            average_dividends = average_dividends['dividends'].mean()
        data_options['yFinance_dividend_yield'] = average_dividends * dividend_payments_each_year /  data_options['last close'].mean()
    
    # keeping collected data
        data_folder = directory + '//' + ticker + ".csv"
        data_options.to_csv(data_folder, index= False )
        
def get_SOFR_curve(current_value):    
    # symbols for tickers: F - January, G - February, H - March, J - April, K - May, M - June, N - Jule, Q - August, U - September, V - October, X - November, Z - December 
    # scrapping and processing market data
    Three_month_SOFR_futures =['SR3G23.CME','SR3H23.CME','SR3J23.CME','SR3K23.CME','SR3M23.CME','SR3N23.CME','SR3Q23.CME','SR3U23.CME','SR3V23.CME','SR3Z23.CME','SR3H24.CME','SR3M24.CME','SR3U24.CME','SR3Z24.CME','SR3H25.CME','SR3M25.CME','SR3U25.CME','SR3Z25.CME','SR3H26.CME','SR3M26.CME','SR3U26.CME','SR3Z26.CME','SR3H27.CME','SR3M27.CME','SR3U27.CME','SR3Z27.CME','SR3H28.CME','SR3M28.CME','SR3U28.CME','SR3Z28.CME']

    Three_month_SOFR_futures_calendar_settlement = [str(datetime.now().date()),'2023-05-17', '2023-06-21','2023-07-19','2023-08-16','2023-09-20','2023-10-18','2023-11-15','2023-12-20','2024-01-17','2024-03-20', '2024-06-20','2024-09-18','2024-12-18','2025-03-19','2025-06-18','2025-09-17','2025-12-17','2026-03-18','2026-06-17','2026-09-16','2026-12-16','2027-03-17','2027-06-16','2027-09-15','2027-12-15', '2028-03-15','2028-06-21','2028-09-20','2028-12-20', '2029-03-21']
    
    
    SOFR_curve = pd.DataFrame()
    for ind,futures in enumerate(Three_month_SOFR_futures):
        data = Ticker(futures)
        futures_SOFR_data = data.history('1d')
        SOFR_curve = pd.concat([SOFR_curve, futures_SOFR_data])
        if futures_SOFR_data.empty:
            Three_month_SOFR_futures_calendar_settlement.pop(ind)
            
    SOFR_curve.reset_index(inplace=True)
    SOFR_curve.drop(['open', 'date', 'high', 'low', 'volume', 'adjclose', 'symbol'], axis=1, inplace=True)
    SOFR_curve['close'] = 100 - SOFR_curve['close']
    SOFR_curve = pd.concat([pd.DataFrame({'close' : [current_value]}), SOFR_curve])
    SOFR_curve['date settlement'] = Three_month_SOFR_futures_calendar_settlement
    SOFR_curve['date settlement'] = SOFR_curve['date settlement'].map(lambda x : datetime.strptime(x,'%Y-%m-%d'))
    return SOFR_curve