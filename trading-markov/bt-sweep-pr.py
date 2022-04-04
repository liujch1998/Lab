ticker = 'DOGE'
trange = '2016-2020'
G = 20

import csv
prices = []
with open('%s-USD_%s.csv' % (ticker, trange)) as f:
    reader = csv.DictReader(f)
    for item in reader:
        if item['Close'] != 'null':
            prices.append(float(item['Close']))
        else:
            print('warning: null found')
all_prices = prices

import numpy as np

invp_ = np.linspace(1/G, 1-1/G, G-1)
r_ = np.linspace(1/G, 1-1/G, G-1)

metrics = ['mu', 'sigma', 'sharpe', 'mdd', 'sterling']
results = { metric : np.zeros((G-1, G-1)) for metric in metrics }

for i, invp in enumerate(invp_):
    for j, r in enumerate(r_):
        p = invp**-1
        s = p**-r
 
        prices = all_prices

        last_price = prices[0]
        shares = (1-r) / last_price
        cash = r # 1 - 1 / (1 + r / (1-r))
        # value = 1

        ret_ = []
        value_ = []
        max_value = 1
        mdd = 0

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

            if len(value_) > 0:
                ret = np.log(value / value_[-1])
                ret_.append(ret)
            value_.append(value)
            max_value = max(max_value, value)
            mdd = max(mdd, np.log(max_value / value))

        mu = np.mean(ret_) * 365 # annualized return
        sigma = np.std(ret_) * 365**0.5 # annualized volatility
        result = {
            'mu': mu,
            'sigma': sigma,
            'sharpe': mu / sigma,
            'mdd': mdd,
            'sterling': mu / mdd,
        }
        for metric in metrics:
            results[metric][j, i] = result[metric]

import matplotlib.pyplot as plt
titles = ['annualized return', 'annualized volatility', 'sharpe', 'mdd', 'sterling']
positions = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]
fig, axs = plt.subplots(2, 3)
for i, metric in enumerate(metrics):
    position = positions[i]
    ax = axs[position[0], position[1]]
    contours = ax.contour(invp_, r_, results[metric])
    ax.clabel(contours, inline=True, fontsize=8)
    ax.set_title(titles[i])
    ax.set_xlabel('1/p')
    ax.set_ylabel('r')
fig.suptitle('Markov strategy backtesting on %s, %s' % (ticker, trange))
plt.show()




'''
mean_cash__, mean_asset__, mean_total__ = [], [], []
std_cash__, std_asset__, std_total__ = [], [], []
for p, r in pairs:
    s = p**(-r)
#    if s*p <= 0.8:
#        mean_cash__.append(-1.0)
#        mean_asset__.append(0.0)
#        mean_total__.append(0.0)
#        std_cash__.append(0.0)
#        std_asset__.append(0.0)
#        std_total__.append(0.0)
#        continue

    cash_, asset_, total_ = [], [], []
#    for _ in range(1000):
#        l = np.random.randint(1, len(prices))
#        i = np.random.randint(0, len(prices)-l)
#        prices_seg = prices[i:(i+l)]
    for i in range(len(prices) - 365):
        prices_seg = prices[i:(i+365)]
        den = prices_seg[0]
        for j in range(len(prices_seg)):
            prices_seg[j] /= den
        mi = 1
        shares = 1
        cash = -1
        for price in prices_seg:
            if s*p <= 1:
                continue
            while price <= mi / p:
                cash -= shares*((1-s)/s) * price
                shares /= s
                mi /= p
            while price >= mi * p**2:
                cash += shares*(1-s**2) * price
                shares *= s**2
                mi *= p**2
        asset = price * shares
        cash_.append(cash)
        asset_.append(asset)
        total_.append(cash + asset)
    mean_cash__.append(np.mean(cash_))
    mean_asset__.append(np.mean(asset_))
    mean_total__.append(np.mean(total_))
    std_cash__.append(np.std(cash_))
    std_asset__.append(np.std(asset_))
    std_total__.append(np.std(total_))

    if s*p > 1:
        upbd = 1 + (1-s)/(s*p-1)
    else:
        upbd = 9999.99
    print('p: %.2f  s: %.2f  r: %.2f  upbd: %.2f  cash: %.2f+-%.2f  asset: %.2f+-%.2f  total: %.2f+-%.2f  sharpe: %.2f' % (p, s, r, upbd, np.mean(cash_), np.std(cash_), np.mean(asset_), np.std(asset_), np.mean(total_), np.std(total_), np.mean(total_) / np.std(total_)))

#    import matplotlib.pyplot as plt
#    plt.plot(cash_)
#    plt.plot(asset_)
#    plt.plot(total_)
#    plt.legend(['cash', 'asset', 'total'])
#    plt.show()

mean_cash__ = np.array(mean_cash__).reshape(len(ps), len(rs)).transpose()
mean_asset__ = np.array(mean_asset__).reshape(len(ps), len(rs)).transpose()
mean_total__ = np.array(mean_total__).reshape(len(ps), len(rs)).transpose()
std_cash__ = np.array(std_cash__).reshape(len(ps), len(rs)).transpose()
std_asset__ = np.array(std_asset__).reshape(len(ps), len(rs)).transpose()
std_total__ = np.array(std_total__).reshape(len(ps), len(rs)).transpose()

sharpe_total__ = mean_total__ / std_total__

import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 3)
contours = axs[0, 0].contour(invps, rs, mean_cash__)
axs[0, 0].clabel(contours, inline=True, fontsize=8)
axs[0, 0].set_title('mean cash')
contours = axs[1, 0].contour(invps, rs, std_cash__)
axs[1, 0].clabel(contours, inline=True, fontsize=8)
axs[1, 0].set_title('std cash')
contours = axs[0, 1].contour(invps, rs, mean_asset__)
axs[0, 1].clabel(contours, inline=True, fontsize=8)
axs[0, 1].set_title('mean asset')
contours = axs[1, 1].contour(invps, rs, std_asset__)
axs[1, 1].clabel(contours, inline=True, fontsize=8)
axs[1, 1].set_title('std asset')
contours = axs[0, 2].contour(invps, rs, mean_total__)
axs[0, 2].clabel(contours, inline=True, fontsize=8)
axs[0, 2].set_title('mean total')
contours = axs[1, 2].contour(invps, rs, std_total__)
axs[1, 2].clabel(contours, inline=True, fontsize=8)
axs[1, 2].set_title('std total')
contours = axs[2, 2].contour(invps, rs, sharpe_total__)
axs[2, 2].clabel(contours, inline=True, fontsize=8)
axs[2, 2].set_title('sharpe total')
for i in range(3):
    for j in range(3):
        axs[i, j].set_xlabel('1/p')
        axs[i, j].set_ylabel('r')
fig.suptitle('Markov strategy backtesting on BTC, 2016-2020, 1-yr window')
plt.show()
'''
