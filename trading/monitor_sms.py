ickers = ['ARKK', 'ARKW', 'ARKQ', 'ARKG']
p = 0.94
from yahoo_fin import stock_info as si
import time
import math
'''
import smtplib, email, re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
def send_email(ticker, count):
    s = smtplib.SMTP('localhost')
    From = 'jl25@illinois.edu'
    To = 'liujch1998@gmail.com'
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f'{ticker} is down to {p}**{count} from last_high'
    msg['From'] = From
    msg['To'] = To
    s.sendmail(From, To, msg.as_string())
    s.quit()
'''
from twilio.rest import Client
client = Client("ACc2359ed1c2f55050e444de0de1ccfe3d", "1862be8a004a4844f5e86059f6d6d6cf")
def send_sms(body):
    global client
    client.messages.create(to="+12173050711", from_="+19292393287", body=body)
last_highs = {}
counts = {}
for ticker in tickers:
    last_highs[ticker] = 0.0
    counts[ticker] = 0
error_count = 0
while True:
    for ticker in tickers:
        try:
            last_high = float(si.get_quote_table(ticker)['52 Week Range'].split(' - ')[1])
            if last_highs[ticker] < last_high:
                last_highs[ticker] = last_high
                counts[ticker] = 0
                print(f'{ticker} new last_high = {last_high}')
        except Exception as e:
            body = f'Error fetching {ticker} last_high: {e}'
            print(body)
            send_sms(body)
            error_count += 1
            if error_count >= 3:
                exit(0)
        try:
            price = si.get_live_price(ticker)
            count = int(math.log(price / last_highs[ticker]) / math.log(p))
            if counts[ticker] < count:
                counts[ticker] = count
                tmp = p ** count
                body = f'{ticker} down to {tmp:.4f} = {p:.2f}**{count}, last_high = {last_highs[ticker]:.2f}, price = {price:.2f}'
                print(body)
                send_sms(body)
        except Exception as e:
            body = f'Error fetching {ticker} price: {e}'
            print(body)
            send_sms(body)
            error_count += 1
            if error_count >= 3:
                exit(0)
    time.sleep(60)
