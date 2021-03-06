import matplotlib.pyplot as plt
import numpy as np

ps_ = [(1.2, 0.9), (1.5, 0.9), (1.5, 0.8), (2.0, 0.8), (2.0, 0.7), (4.0, 0.49)]
ps_ += [(1.0001, 0.99995)]

# bull and bear
fig, axs = plt.subplots(1, 3)
## bull
for p, s in ps_:
    def bull(peak):
        k = int(np.log(peak) / np.log(p))
        revenue = (1-s) * p * ((s*p)**k-1) / (s*p-1)
        return revenue
    logx_ = np.linspace(0.0, 7.0, 1001)
    x_ = np.exp(logx_)
    y_ = np.vectorize(bull)(x_)
    axs[0].plot(x_, y_)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel('peak price')
axs[0].set_ylabel('revenue')
axs[0].set_title('Revenue in Bull Market')
axs[0].legend(ps_)
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
axs[1].legend(ps_)
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
axs[2].legend(ps_)
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

