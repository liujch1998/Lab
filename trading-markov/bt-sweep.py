import csv
prices = []
with open('BTC-USD_0204-0519.csv') as f:
    reader = csv.DictReader(f)
    for item in reader:
        if item['Close'] != 'null':
            prices.append(float(item['Close']))
        else:
            print('warning: null found')

import numpy as np
#M = 50
#invps = np.linspace(1/M, 1-1/M, M-1)
#ss = np.linspace(1/M, 1-1/M, M-1)
M = 50
invps = np.linspace(0.5, 0.99, M)
ss = np.linspace(0.5, 0.99, M)
ps = invps**(-1)
pairs = [(p, s) for p in ps for s in ss]
mean_cash__, mean_asset__, mean_total__ = [], [], []
std_cash__, std_asset__, std_total__ = [], [], []
for p, s in pairs:
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
    for i in range(len(prices) - 91):
        prices_seg = prices[i:(i+91)]
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
            while price >= mi * p:
                cash += shares*(1-s) * price
                shares *= s
                mi *= p
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
    print('p: %.2f  s: %.2f  upbd: %.2f  cash: %.2f+-%.2f  asset: %.2f+-%.2f  total: %.2f+-%.2f  sharpe: %.2f' % (p, s, upbd, np.mean(cash_), np.std(cash_), np.mean(asset_), np.std(asset_), np.mean(total_), np.std(total_), np.mean(total_) / np.std(total_)))

#    import matplotlib.pyplot as plt
#    plt.plot(cash_)
#    plt.plot(asset_)
#    plt.plot(total_)
#    plt.legend(['cash', 'asset', 'total'])
#    plt.show()

mean_cash__ = np.array(mean_cash__).reshape(len(ps), len(ss)).transpose()
mean_asset__ = np.array(mean_asset__).reshape(len(ps), len(ss)).transpose()
mean_total__ = np.array(mean_total__).reshape(len(ps), len(ss)).transpose()
std_cash__ = np.array(std_cash__).reshape(len(ps), len(ss)).transpose()
std_asset__ = np.array(std_asset__).reshape(len(ps), len(ss)).transpose()
std_total__ = np.array(std_total__).reshape(len(ps), len(ss)).transpose()

sharpe_total__ = mean_total__ / std_total__

import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 3)
contours = axs[0, 0].contour(invps, ss, mean_cash__)
axs[0, 0].clabel(contours, inline=True, fontsize=8)
#axs[0, 0].plot(invps, ss, color='red')
axs[0, 0].set_title('mean cash')
contours = axs[1, 0].contour(invps, ss, std_cash__)
axs[1, 0].clabel(contours, inline=True, fontsize=8)
axs[1, 0].set_title('std cash')
contours = axs[0, 1].contour(invps, ss, mean_asset__)
axs[0, 1].clabel(contours, inline=True, fontsize=8)
axs[0, 1].set_title('mean asset')
contours = axs[1, 1].contour(invps, ss, std_asset__)
axs[1, 1].clabel(contours, inline=True, fontsize=8)
axs[1, 1].set_title('std asset')
contours = axs[0, 2].contour(invps, ss, mean_total__)
axs[0, 2].clabel(contours, inline=True, fontsize=8)
axs[0, 2].set_title('mean total')
contours = axs[1, 2].contour(invps, ss, std_total__)
axs[1, 2].clabel(contours, inline=True, fontsize=8)
axs[1, 2].set_title('std total')
contours = axs[2, 2].contour(invps, ss, sharpe_total__)
axs[2, 2].clabel(contours, inline=True, fontsize=8)
axs[2, 2].set_title('sharpe total')
for i in range(3):
    for j in range(3):
        axs[i, j].set_xlabel('1/p')
        axs[i, j].set_ylabel('s')
fig.suptitle('Markov strategy backtesting on BTC, 2016-2020, 1-yr window')
plt.show()

