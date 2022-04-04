import matplotlib.pyplot as plt
import numpy as np

ps_ = [(1.2, 0.9), (1.5, 0.9), (1.5, 0.8), (2.0, 0.8), (2.0, 0.7), (4.0, 0.49)]
ps_ += [(1.0001, 0.99995)]

# const r
r = 0.5
ps_ = [
    (1.01, 1.01**-r),
    (1.1, 1.1**-r),
    (1.3, 1.3**-r),
    (2.0, 2.0**-r),
    (5.0, 5.0**-r),
]

# vary r, small p (limit)
p = 1.01
ps_ = [
    (p, p**-0.99),
    (p, p**-0.9),
    (p, p**-0.8),
    (p, p**-0.5),
    (p, p**-0.2),
    (p, p**-0.1),
]

# bull and bear
fig, axs = plt.subplots(1, 3)
## bull
for p, s in ps_:
    def bull(peak):
        k = int(np.log(peak) / np.log(p))
        revenue = (1-s) * p * ((s*p)**k-1) / (s*p-1)
        asset = s**k * peak
        total = revenue + asset
        return revenue, asset, total
    logx_ = np.linspace(0.0, 7.0, 1001)
    x_ = np.exp(logx_)
    revenue_, asset_, total_ = np.vectorize(bull)(x_)
    axs[0].plot(x_, revenue_)
    axs[1].plot(x_, asset_)
    axs[2].plot(x_, total_)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel('peak price')
axs[0].set_ylabel('revenue')
axs[0].set_title('Revenue in Bull Market')
axs[0].legend(['p=%.2f, s=%.3f, r=%.2f' % (p, s, -np.log(s)/np.log(p)) for p, s in ps_])
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[2].set_xscale('log')
axs[2].set_yscale('log')
'''
## bear
for p, s in ps_:
    def bear(bottom):
        k = int(-np.log(bottom) / np.log(p))
        cost = 1 + (1-s) * (1-(s*p)**-k) / (s*p-1)
        return cost
    logx_ = np.linspace(-7.0, 0.0, 1001)
    x_ = np.exp(logx_)
    y_ = np.vectorize(bear)(x_)
    axs[1].plot(x_, y_)
axs[1].set_xscale('log')
#axs[1].set_yscale('log')
axs[1].set_xlabel('bottom price')
axs[1].set_ylabel('total cost')
axs[1].set_title('Total Cost in Bear Market')
axs[1].legend(['p=%.2f, s=%.3f, r=%.2f' % (p, s, -np.log(s)/np.log(p)) for p, s in ps_])
for p, s in ps_:
    def bear(bottom):
        k = int(-np.log(bottom) / np.log(p))
        cost = 1 + (1-s) * (1-(s*p)**-k) / (s*p-1)
        avg_cost = cost / s**-k
        return avg_cost
    logx_ = np.linspace(-7.0, 0.0, 1001)
    x_ = np.exp(logx_)
    y_ = np.vectorize(bear)(x_)
    axs[2].plot(x_, y_)
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_xlabel('bottom price')
axs[2].set_ylabel('avg cost')
axs[2].set_title('Avg Cost in Bear Market')
axs[2].legend(['p=%.2f, s=%.3f, r=%.2f' % (p, s, -np.log(s)/np.log(p)) for p, s in ps_])
'''
plt.show()

# bump and dip
fig, axs = plt.subplots(1, 2)
## bump
for p, s in ps_:
    def bump(peak):
        k = int(np.log(peak) / np.log(p))
        revenue = (1-s) * (p-1) * ((s*p)**k-1) / (s*p-1)
        return revenue
    logx_ = np.linspace(0.0, 7.0, 1001)
    x_ = np.exp(logx_)
    y_ = np.vectorize(bump)(x_)
    axs[0].plot(x_, y_)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel('peak price')
axs[0].set_ylabel('revenue')
axs[0].set_title('Revenue in Market Bump')
axs[0].legend(ps_)
## dip
for p, s in ps_:
    def dip(bottom):
        k = int(-np.log(bottom) / np.log(p))
        revenue = (1-s) * (p-1) * (1-(s*p)**-k) / (s*p-1)
        return revenue
    logx_ = np.linspace(-7.0, 0.0, 1001)
    x_ = np.exp(logx_)
    y_ = np.vectorize(dip)(x_)
    axs[1].plot(x_, y_)
axs[1].set_xscale('log')
#axs[1].set_yscale('log')
axs[1].set_xlabel('bottom price')
axs[1].set_ylabel('revenue')
axs[1].set_title('Revenue in Market Dip')
axs[1].legend(ps_)
plt.show()

