import robin_stocks as rs
from time import sleep
from getpass import getpass

ticker = 'NIO'
p = 0.94
c = 1

rs.login(username='liujch1998',
         password=None,#getpass(),
         expiresIn=86400 * 365,
         by_sms=True)

try:
    fund = rs.stocks.get_fundamentals(ticker, info=None)
    last_high = float(fund[0]['high_52_weeks'])
    print('last_high = ', last_high)
except Exception as e:
    print('Error fetching fundamentals: ', e)
    exit(0)

next_buy = p * last_high
with open('last_buy.txt', 'r') as f:
    lines = f.readlines()
    if lines != []:
        last_buy = float(lines[-1].strip('\n'))
        next_buy = p * last_buy
        print('last_buy = ', last_buy)
print('next_buy = ', next_buy)

while True:
    try:
        price = rs.stocks.get_latest_price(ticker, includeExtendedHours=True)
        price = float(price[0])
        if last_high < price:
            last_high = price
            next_buy = p * last_high
        if price <= next_buy:
            try:
                buy = rs.orders.order_buy_fractional_by_price(ticker, c, extendedHours=True)
                print('Bought at price = ', price)
                with open('last_buy.txt', 'a') as f:
                    f.write(str(next_buy) + '\n')
                next_buy *= p
                print('next_buy = ', next_buy)
            except Exception as e:
                print('Error placing order: ', e)
    except Exception as e:
        print('Error fetching current price: ', e)
    sleep(60)

