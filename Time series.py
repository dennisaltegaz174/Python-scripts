# Finding the order of  differencing (d) in ARIMA model
from statsmodels.tsa.stattools import adfuller
from numpy import log
result=adfuller(df.value.dropna())
print('ADF Statistic: %f'% result[0])
print('p-value: %f' %result[1])

# Since the p-value is greater than the significance level, let's difference the  series and see how the plot looks like
import numpy  as  np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf,plot_acf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7),'figure.dpi':120})
# Data importation
df =  pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv',names=['value'],header=0)

# Original series
fig,axes = plt.subplots(3,2, sharex=True)
axes[0,0].plot(df.value);
axes[0,0].plot(df.value);
axes[0,0].set_title('Original Series')
plot_acf(df.value,ax=axes[0,1])

# Fist differencing
axes[1,0].plot(df.value.diff());
axes[1,0].set_title('1st Order Differencing')
plot_acf(df.value.diff().dropna(),ax=axes[1,1])

# 2nd differencing
axes[2,0].plot(df.value.diff().diff());
axes[2,0].set_title('2nd  Order Differencing')
plot_acf(df.value.diff().diff().dropna(),ax=axes[2,1])
plt.show()