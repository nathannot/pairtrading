import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt

st.title('Pairs Trading for Popular US Stocks')

options = st.selectbox('Pick trading pair to analyse',
                       ('Amazon vs Walmart', 'Nvidia vs AMD','Apple vs Microsoft', 'Coke vs Pepsi', 'Netflix vs Disney'))

mappings = {'Amazon vs Walmart':['AMZN','WMT'], 'Nvidia vs AMD':['NVDA','AMD'],'Apple vs Microsoft':['AAPL', 'MSFT'],
             'Coke vs Pepsi':['KO', 'PEP'], 'Netflix vs Disney':['NFLX', 'DIS']}
tickers = mappings[options]
date_start = st.date_input('Enter start Date', min_value = datetime.date(2010,1,1), value = datetime.date(2016,1,1))
date_end = st.date_input('Enter Finish Date', min_value = datetime.date(2010,1,1), value = datetime.date(2021,1,1))
st.write('Make sure start date is before end date or you will get an error')
start_date = date_start
end_date = date_end
data = yf.download(tickers, start = start_date, end = end_date, progress=False)['Adj Close']
x = data.iloc[:,0]
y = data.iloc[:,1]
fig, ax = plt.subplots()
ax.plot(x.index, x, label=f'{tickers[0]}')
ax.legend(loc='upper left')
ax1 = ax.twinx()
ax1.plot(y.index, y, color='r', label = f'{tickers[1]}')
ax1.legend(loc='lower right')
ax.set_title(f'Daily chart of {tickers[0]} vs {tickers[1]}')
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
st.pyplot(fig)

nx = len(x)
ny = len(y)
xbar = (1/nx)*np.sum(x)
ybar = (1/ny)*np.sum(y)
sx2 = np.sum(x**2)-nx*(xbar)**2
sxy = np.sum(x*y)-nx*xbar*ybar
beta = sxy/sx2
alpha = ybar-beta*xbar
yhat = alpha+beta*x
fig1, ax3 = plt.subplots()
ax3.scatter(x,y, alpha = 0.5)
ax3.plot(x,yhat, 'r')
ax3.set_title(f'Scatter plot of {tickers[0]} vs {tickers[1]}')
st.pyplot(fig1)

s = y - beta*x
s_mean = (1/len(s))*np.sum(s)
s_std = np.sqrt((1/(len(s)-1))*(np.sum(s**2)-len(s)*s_mean**2))
s_z = (s-s_mean)/s_std

st.subheader('Define your risk-reward trading parameters below')
st.write('If unsure of what they mean leave the default values')
trading_range = st.slider('Select trading risk',min_value = 1.5, max_value=2.5,value=2.0)
st.write('1.5 indicates higher risk, 2.5 indicates lower risk')
risk = st.slider('Select your stop loss level', min_value = 0.2, max_value = 2.0, value=0.5)
risk_reward = st.slider('Select your desired risk to reward', min_value = 1.0, max_value = 4.0, value=2.0)

short = s_z[s_z>trading_range]
long = s_z[s_z<-trading_range]
fig2, ax4 = plt.subplots()
ax4.plot(x.index, s_z)
ax4.plot(x.index, nx*[trading_range], color='orange', linestyle='--',label=f'short signal for {tickers[1]}')
ax4.plot(x.index, nx*[trading_range+risk], color='red',linestyle='--',label=f'stop loss for {tickers[1]}')
ax4.plot(x.index, nx*[max(trading_range-risk*risk_reward,0)], color='green',linestyle='--',label=f'take profit for {tickers[1]}')
ax4.plot(x.index, nx*[-trading_range], color='purple', linestyle='--',label=f'buy signal for {tickers[1]}')
ax4.plot(x.index, nx*[-trading_range-0.5], color='red',linestyle='--')
ax4.plot(x.index, nx*[min(-(trading_range-risk*risk_reward),0)], color='green',linestyle='--')
ax4.legend(bbox_to_anchor = (1.05,1))
ax4.scatter(short.index,short,c='r',marker='o', s=10)
ax4.scatter(long.index,long,c='r',marker='o', s=10)
ax4.set_title('Signals chart')
for tick in ax4.get_xticklabels():
    tick.set_rotation(45)
st.pyplot(fig2)


dates_sell = s_z[(s_z>=trading_range-0.05) & (s_z <=trading_range+0.1)].index
dates_buy = s_z[(s_z<=-trading_range+0.05) & (s_z >=-trading_range -0.1)].index
dates_stoploss = s_z[(s_z<=-trading_range-risk) | (s_z >=trading_range+risk)].index
dates_tp = s_z[(s_z<=0.1 -trading_range+risk*risk_reward) & (s_z >=-trading_range+risk*risk_reward-0.05) | (s_z>=trading_range-risk*risk_reward-0.1) & (s_z <=trading_range-risk*risk_reward+0.05)].index

initial_portfolio = 1
portfolio = initial_portfolio
positions = {tickers[0]:0, tickers[1]:0}
cash = portfolio

data['signal'] = 0
data.loc[dates_buy,'signal'] = 1
data.loc[dates_sell,'signal'] = -1
data.loc[dates_stoploss,'signal'] = 'stop loss'
data.loc[dates_tp, 'signal'] = 'take profit'

portfolio_changes = []
for date, row in data.iterrows():
    price1 = row[tickers[0]]
    price2 = row[tickers[1]]
    signal = row['signal']

    if signal == 1:
        positions[tickers[0]] -= round(beta)
        positions[tickers[1]] += 1
        cash += price1*round(beta) - price2
    elif signal == -1:
        positions[tickers[0]] += round(beta)
        positions[tickers[1]] -= 1
        cash += price2 - price1*round(beta)
    elif signal == 'stop loss':
        cash += positions[tickers[0]]*price1+positions[tickers[1]]*price2
        positions = {tickers[0]:0, tickers[1]:0}
    elif signal == 'take profit':
        cash += positions[tickers[0]]*price1 + positions[tickers[1]]*price2
        positions = {tickers[0]:0, tickers[1]:0}
                                                           
    portfolio = cash+positions[tickers[0]]*price1+positions[tickers[1]]*price2
    portfolio_changes.append({'Date':date, 'Portfolio':portfolio})

port_df = pd.DataFrame(portfolio_changes).set_index('Date')
fig3, ax5 = plt.subplots()
ax5.plot(port_df.index, port_df)
ax5.set_title(f'Profit per share of {tickers[1]}')
for tick in ax5.get_xticklabels():
    tick.set_rotation(45)
st.pyplot(fig3)
st.write('Total profit is profit per share x number of shares bought')
st.write('This is version 1 of the app')
