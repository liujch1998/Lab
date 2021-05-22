import numpy as np
import matplotlib.pyplot as plt

ps = [1.1, 1.618, 2.618]
p = 1.1

def const_share(bot):
    k = int(-np.log(bot) / np.log(p))
    avg_cost = (1 - p**(-(k+1))) / ((k+1) * (1 - p**-1))
    return avg_cost

def const_cost(bot):
    k = int(-np.log(bot) / np.log(p))
    avg_cost = ((k+1) * (p-1)) / (p**(k+1) - 1)
    return avg_cost

def mixed(bot):
    k = int(-np.log(bot) / np.log(p))
    avg_cost = p**(-k/2)
    return avg_cost

logbots = np.linspace(-10.0, 0.0, 100)
bots = np.exp(logbots)

for f in [const_share, const_cost, mixed]:
    y = np.vectorize(f)(bots)
    plt.plot(bots, y)
plt.xscale('log')
plt.yscale('log')
plt.show()
