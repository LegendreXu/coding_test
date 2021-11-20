import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Data Processing
data  = pd.read_csv('data.csv')
data_id = data.set_index(['ticker','date'])
data_last = data_id['last'].unstack().transpose()
data_volume = data_id['volume'].unstack().transpose()

# Parameters Setting
initial_budget = 1000000
n_stocks = 10
trading_cost = 0
lookback = 120
rebalance = 20
risk_aversion = 1

net_value = pd.DataFrame([0]*data_last.shape[0], index=data_last.index, columns=['net_value'])
cash = pd.DataFrame([0]*data_last.shape[0], index=data_last.index,columns=['cash'])
# Target Weight
weight = pd.DataFrame(np.zeros(shape=data_last.shape),columns=data_last.columns, index=data_last.index)
# Practical Position
stock_pos = pd.DataFrame(np.zeros(shape=data_last.shape),columns=data_last.columns, index=data_last.index)



for i in range(0,lookback):
    cash.iloc[i] = initial_budget
    net_value.iloc[i] = initial_budget


for i in range(lookback, data_last.shape[0]):
    today_price = data_last.iloc[i-1:i+1]
    today_ret = (np.log(today_price)).diff()[1:]
    today_ret = today_ret.fillna(0)
    if (i-lookback)%20 == 0:
        temp = data_last.iloc[i-lookback:i-1].dropna(axis=1)
        temp_ret = (np.log(temp)).diff()[1:]
        ret = temp_ret.sum(axis=0)
        vol = temp_ret.std(axis=0)
        momentum =  ret - risk_aversion * vol
        idx_buy = momentum.nlargest(n_stocks).index
        stock_buy = data_last.iloc[i][momentum.nlargest(n_stocks).index]
        weight.iloc[i][idx_buy] = 1.0/n_stocks
        stock_pos.iloc[i] = weight.iloc[i]*net_value.iloc[i-1].values[0]*(today_ret[stock_pos.columns]+1)

        cash.iloc[i] = (1-sum(weight.iloc[i]))*net_value.iloc[i-1].values[0]
    else:
        weight.iloc[i] = weight.iloc[i-1]
        stock_pos.iloc[i] = stock_pos.iloc[i-1]*(today_ret[stock_pos.columns]+1)
        cash.iloc[i] = cash.iloc[i-1]
    net_value.iloc[i] = cash.iloc[i] + sum(stock_pos.iloc[i])

ret = np.log(net_value).diff().dropna()

annual_ret = ret.mean().values[0]*252
annual_vol = ret.std().values[0]*np.sqrt(252)
sharpe = annual_ret/annual_vol
s=(ret+1).cumprod()
mdd_pct = (1-np.ptp(s)/s.max()).values[0]
mdd_usd = (net_value.cummax()-net_value).max()
mdd_usd = mdd_usd.values[0]
r_d = ret[ret<0].dropna()
down_deviation = (np.sqrt((ret.mean().values[0]- r_d)**2).sum())/r_d.shape[0]
down_deviation = down_deviation.values[0]
sortino = annual_ret/ down_deviation

print("Annualized Return: ", annual_ret*100, "%")
print("Annualized Volatility: ", annual_vol*100, "%")
print("Downside Deviation: ", down_deviation*100, "%")
print("Max Drawdown(in percentage): ", mdd_pct*100, "%")
print("Max Drawdown(in dollars): ", mdd_usd)
print("Sharpe Ratio: ", sharpe)
print("Sortino Ratio: ", sortino)


fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(111)
i = np.argmax(np.maximum.accumulate(net_value) - net_value) # end of the period
j = np.argmax(net_value[:i]) # start of period
ax.plot(net_value)
ax.plot([net_value.index[i], net_value.index[j]], [net_value.iloc[i], net_value.iloc[j]], 'o', color='Red', markersize=10)
tick_space = 126
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_space))
plt.grid()
plt.show()


