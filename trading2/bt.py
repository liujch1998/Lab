import csv
prices = []
with open('BTC-USD_0204-0519.csv') as f:
    reader = csv.DictReader(f)
    for item in reader:
        if item['Close'] != 'null':
            prices.append(float(item['Close']))
        else:
            print('warning: null found')
prices = prices[8:]

import numpy as np
invp = 0.89#0.79
s = 0.91#0.80
p = invp**(-1)
cash_, asset_, total_ = [], [], []
mi = prices[0]
shares = 1000 / prices[0]
cash = -1000
for price in prices[:91]:
    while price <= mi / p:
        cash -= shares*((1-s)/s) * price
        shares /= s
        mi /= p
    while price >= mi * p:
        cash += shares*(1-s) * price
        shares *= s
        mi *= p
    asset = price * shares
    cash_.append(cash)
    asset_.append(asset)
    total_.append(cash + asset)

import matplotlib.pyplot as plt
plt.plot(cash_)
plt.plot(asset_)
plt.plot(total_)
plt.legend(['cash', 'asset', 'total'])
plt.show()
