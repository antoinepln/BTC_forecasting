import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import pandas_datareader as pdr
import ipdb

load_dotenv()

API_KEY = os.getenv('GLASSNODES_API_KEY')

def get_btc_OHLC():
    url_glassnode = 'https://api.glassnode.com/v1/metrics/market/price_usd_ohlc'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    close = []
    high = []
    low = []
    Open = []
    date = []
    for x in range(len(data)) :
        close.append(data[x]['o']['c'])
        high.append(data[x]['o']['h'])
        low.append(data[x]['o']['l'])
        Open.append(data[x]['o']['o'])
        date.append(data[x]['t'])
    dict_df = {
        'date' : date,
        'close' : close,
        'open' : Open,
        'high' : high,
        'low' : low
    }
    df = pd.DataFrame(dict_df)
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def get_drawdown_ath():
    url_glassnode = 'https://api.glassnode.com/v1/metrics/market/price_drawdown_relative'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    date = []
    price = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        price.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['price'] = price
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df + 1



def get_rsi(df,n):
    n = 14
    df_rsi = df.copy()
    def rma(x, n, y0):
        a = (n-1) / n
        ak = a**np.arange(len(x)-1, -1, -1)
        return np.append(y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1))
    df_rsi.reset_index(inplace = True)
    df_rsi['change'] = df_rsi['close'].diff()
    df_rsi['gain'] = df_rsi.change.mask(df_rsi.change < 0, 0.0)
    df_rsi['loss'] = -df_rsi.change.mask(df_rsi.change > 0, -0.0)
    df_rsi.loc[n:,'avg_gain'] = rma( df_rsi.gain[n+1:].values, n, df_rsi.loc[:n, 'gain'].mean())
    df_rsi.loc[n:,'avg_loss'] = rma( df_rsi.loss[n+1:].values, n, df_rsi.loc[:n, 'loss'].mean())
    df_rsi['rs'] = df_rsi.avg_gain / df_rsi.avg_loss
    df_rsi['rsi_14'] = 100 - (100 / (1 + df_rsi.rs))
    df_rsi.set_index('date', inplace = True)
    return df_rsi['rsi_14']


def get_EMA_MACD(df):

    ema_10 = df['close'].ewm(span = 10).mean()

    ema_12 = df['close'].ewm(span = 12).mean()
    ema_26 = df['close'].ewm(span = 26).mean()

    macd = ema_12 - ema_26

    return macd, ema_10




def get_SMI(df) :
    period = 10
    df_smi = df.copy()
    Max= df.iloc[:,0].rolling(10).max()
    Min = df.iloc[:,0].rolling(10).min()
    Mean = (Min + Max )/2
    h = df_smi['close'] - Mean
    HS2 = h.ewm(span = 3).mean().ewm(span = 3).mean()
    DHL2  = (Max -Min).ewm(span = 3).mean().ewm(span = 3).mean()
    return HS2 / DHL2

def get_CCI(df)  :
    df_cci = df.copy()
    cours_moyen = (df['close'] + df['high'] + df['low']) / 3
    SMATP = cours_moyen.rolling(window=20).mean()
    deviation_moyenne  = abs(SMATP - cours_moyen).rolling(window=20).mean()
    CCI = (cours_moyen - SMATP) / (0.015*deviation_moyenne)
    return CCI


def get_william_A_D(df) :
    df_a_d = df.copy()
    TRH = {}
    for x in range(1,len(df_a_d)) :
        if df_a_d['close'][x-1] > df_a_d['high'][x] :
            TRH[df_a_d.index[x]] = df_a_d['close'][x-1]
        else :
            TRH[df_a_d.index[x]] = df_a_d['high'][x]
    TRL = {}
    for x in range(1,len(df)) :
        if df_a_d['close'][x-1] < df_a_d['low'][x] :
            TRL[df_a_d.index[x]] = df_a_d['close'][x-1]
        else :
            TRL[df_a_d.index[x]] = df_a_d['low'][x]
    A_D = {}
    for x in range(len(df_a_d)):
        if x == 0 :
            A_D[df_a_d.index[x]] = 0
        elif df_a_d['close'][x] > df_a_d['close'][x-1] :
            A_D[df_a_d.index[x]] = df_a_d['close'][x] - TRL[df_a_d.index[x]]
        elif df_a_d['close'][x] < df_a_d['close'][x-1] :
            A_D[df_a_d.index[x]] = df_a_d['close'][x] - TRH[df_a_d.index[x]]
        else :
            A_D[df_a_d.index[x]] = 0
    df_a_d['A_D'] = A_D.values()
    df_a_d['williams_A_D'] = df_a_d['A_D'].cumsum()
    return df_a_d['williams_A_D']


def get_ATR(df):
    df_atr = df.copy()
    a = df_atr['high'] - df_atr['low']
    b = abs(df_atr['high'] - df_atr['close'])
    c = abs(df_atr['low'] -df_atr['close'])
    TR = []
    for x in range(len(df_atr)):
        TR.append(max(a[x],b[x],c[x]))
    df_atr['TR'] = TR
    df_atr['ATR'] = df_atr['TR'].rolling(window=14).mean()
    return df_atr['ATR']

def new_adress() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/addresses/new_non_zero_count'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    date = []
    adresses = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        adresses.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['adresses'] = adresses
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def sopr() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/indicators/sopr'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    date = []
    sopr = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        sopr.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['sopr'] = sopr
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def comp_ribbon() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/indicators/difficulty_ribbon_compression'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    date = []
    comp_ribbon = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        comp_ribbon.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['comp_ribbon'] = comp_ribbon
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def create_utxo() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/blockchain/utxo_created_count'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    date = []
    utxo = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        utxo.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['utxo'] = utxo
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def transac_sec() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/transactions/rate'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    date = []
    transac = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        transac.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['transac'] = transac
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def mvrv_z() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/market/mvrv_z_score'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    date = []
    mvrv_z = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        mvrv_z.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['mvrv_z'] = mvrv_z
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def nvts() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/indicators/nvts'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    date = []
    nvts = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        nvts.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['nvts'] = nvts
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def pct_profit() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/supply/profit_relative'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    date = []
    profit = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        profit.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['profit'] = profit
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def supp_last_act() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/supply/active_1d_1w'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    supp_last_act = []
    date = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        supp_last_act.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['supp_last_act'] = supp_last_act
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def active_adress() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/addresses/active_count'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    active_adress = []
    date = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        active_adress.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['active_adress'] = active_adress
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def fees_total() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/fees/volume_sum'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    fees_total = []
    date = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        fees_total.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['fees_total'] = fees_total
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df


def hash_rate() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/mining/hash_rate_mean'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    hash_rate = []
    date = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        hash_rate.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['hash_rate'] = hash_rate
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def transactions_count() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/transactions/count'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    transactions_count = []
    date = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        transactions_count.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['transactions_count'] = transactions_count
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def utxo_spent() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/blockchain/utxo_spent_count'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    utxo_spent = []
    date = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        utxo_spent.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['utxo_spent'] = utxo_spent
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def utxo_create_tot() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/blockchain/utxo_created_value_sum'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    utxo_create_tot = []
    date = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        utxo_create_tot.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['utxo_create_tot'] = utxo_create_tot
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df

def utxo_spent_tot() :
    url_glassnode = 'https://api.glassnode.com/v1/metrics/blockchain/utxo_spent_value_sum'
    parameters = {
     'api_key' : API_KEY,
     'a' : 'BTC'
             }
    data = requests.get(url  = url_glassnode, params = parameters).json()
    utxo_spent_tot = []
    date = []
    for i in range(len(data)) :
        date.append(data[i]['t'])
        utxo_spent_tot.append(data[i]['v'])
    df = pd.DataFrame(date, columns = ['date'])
    df['utxo_spent_tot'] = utxo_spent_tot
    df['date'] = pd.to_datetime(df['date'], unit = 's')
    df.set_index('date', inplace = True)
    return df


def get_sp(df_) :
    df = pdr.data.DataReader("^gspc",
                       start='2010-07-19',
                       data_source='yahoo')
    df.reset_index(inplace = True)
    df['Date'] = pd.to_datetime(df['Date'])

    price = [df['Close'][0],df['Close'][0], df['Close'][0]]
    for i in range(1,len(df)) :
        if (df['Date'][i] - df['Date'][i-1]).days == 1 :
            price.append(df['Close'][i])
        elif (df['Date'][i] - df['Date'][i-1]).days == 2 :
            price.append(df['Close'][i-1])
            price.append(df['Close'][i])
        elif (df['Date'][i] - df['Date'][i-1]).days == 3 :
            price.append(df['Close'][i-1])
            price.append(df['Close'][i-1])
            price.append(df['Close'][i])
        elif (df['Date'][i] - df['Date'][i-1]).days == 4 :
            price.append(df['Close'][i-1])
            price.append(df['Close'][i-1])
            price.append(df['Close'][i-1])
            price.append(df['Close'][i])
        elif (df['Date'][i] - df['Date'][i-1]).days == 5 :
            price.append(df['Close'][i-1])
            price.append(df['Close'][i-1])
            price.append(df['Close'][i-1])
            price.append(df['Close'][i-1])
            price.append(df['Close'][i])
    price.append(df['Close'].iloc[-1])
    #price.append(df['Close'].iloc[-1])
    #price.append(df['Close'].iloc[-1])
    df_sp = df_.copy()
    df_sp['sp'] = price

    pct = [0]
    for x in range(1,len(df_sp)) :
        mouv = (df_sp['sp'].iloc[x] - df_sp['sp'].iloc[x-1])/ df_sp['sp'].iloc[x-1]
        pct.append(mouv)

    df_sp['sp'] = pct

    return df_sp['sp']



df = get_btc_OHLC()
df['rsi_14']= get_rsi(df,14)
df['macd'], df['ema_10'] = get_MACD(df)
df['SMI'] = get_SMI(df)
df['CCI'] = get_CCI(df)
df['william_a/d'] = get_william_A_D(df)
df['ATR'] = get_ATR(df)
df['ath'] = get_drawdown_ath()
df['new_adress'] = new_adress()
df['sopr'] = sopr()
df['comp_ribbon'] = comp_ribbon()
df['utxo'] = create_utxo()
df['transac_sec'] = transac_sec()
df['mvrv_z'] = mvrv_z()
df['nvts'] = nvts()
df['pct_profit'] = pct_profit()
df['supp_last_act'] = supp_last_act()
df['sp'] = get_sp(df)
df['active_adress'] = active_adress()
df['fees_total'] = fees_total()
df['hash_rate'] = hash_rate()
df['transactions_count'] = transactions_count()
df['utxo_spent'] = utxo_spent()








#df.to_csv('../data/data.csv')

if __name__ == '__main__':
    df.to_csv('data/data.csv')

