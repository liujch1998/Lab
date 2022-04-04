ticker = 'BTC'
trange = '2016-2020'

import csv
prices = []
with open('%s-USD_%s.csv' % (ticker, trange)) as f:
    reader = csv.DictReader(f)
    for item in reader:
        if item['Close'] != 'null':
            prices.append(float(item['Close']))
        else:
            print('warning: null found')
prices = prices[730:]

import numpy as np
invp = 0.618
r = 0.50
p = invp**(-1)
s = p**-r
cash_, asset_, value_ = [], [], []
last_price = prices[0]
shares = (1-r) / last_price
cash = r
for price in prices:
    while price <= last_price / p:
        cash -= shares*((1-s)/s) * (last_price / p)
        shares /= s
        last_price /= p
    while price >= last_price * p:
        cash += shares*(1-s) * (last_price * p)
        shares *= s
        last_price *= p
    asset = price * shares
    value = cash + asset
    cash_.append(cash)
    asset_.append(asset)
    value_.append(value)

import matplotlib.pyplot as plt
plt.plot(cash_)
plt.plot(asset_)
plt.plot(value_)
plt.legend(['cash', 'asset', 'value'])
plt.show()
