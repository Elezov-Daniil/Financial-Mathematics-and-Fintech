import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


def scatter_3D(ticker, data_dict):
    plt.figure(figsize = (8,8))
    ax = plt.axes(projection ="3d")

    for k in data_dict["implied_volatility_surface"]:
        ax.scatter( k['expiry_date_in_act365_year_fraction'], np.array(k['strikes'])/k['reference_forward'], k['mid_implied_volatilities'], s=10)

    ax.set_title(ticker)
    ax.set_xlabel(r'Maturity (year fraction)')
    ax.set_ylabel(r'Strike (fwd moneyness)')
    ax.set_zlabel(r'Implied Volatility')
    ax.grid()
    ax.view_init(20, 40)
    plt.show()

def lines_3D(ticker, data_dict):
    plt.figure(figsize = (8,8))
    ax = plt.axes(projection ="3d")

    for k in data_dict["implied_volatility_surface"]:
        ax.plot(np.array(k['strikes'])/k['reference_forward'], k['mid_implied_volatilities'], zs=k['expiry_date_in_act365_year_fraction'],zdir='x')
        ax.scatter( k['expiry_date_in_act365_year_fraction'], np.array(k['strikes'])/k['reference_forward'], k['mid_implied_volatilities'], s=10)

    ax.set_title(ticker)
    ax.set_xlabel(r'Maturity (year fraction)')
    ax.axes.set_xlim3d(left=0, right=2.5)
    ax.set_ylabel(r'Strike (fwd moneyness)')
    ax.axes.set_ylim3d(bottom=0.4, top=2)
    ax.set_zlabel(r'Implied Volatility')
    ax.axes.set_zlim3d(bottom=0.0, top=1) 
    ax.set_zticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid()
    ax.view_init(20, 40)
    plt.show()

def lines_2D(ticker, data_dict):
    plt.figure(figsize = (8,8))
    ax = plt.gca()

    for k in data_dict["implied_volatility_surface"]:
        ax.scatter(np.array(k['strikes'])/k['reference_forward'], k['mid_implied_volatilities'], label= k['expiry_date'], s=10)
        plt.plot(np.array(k['strikes'])/k['reference_forward'],k['mid_implied_volatilities'])

    ax.set_title(ticker)
    ax.set_xlabel(r'strike (fwd moneyness)')
    ax.set_ylabel(r'Implied Volatility')
    ax.legend()
    ax.grid()
    plt.show()