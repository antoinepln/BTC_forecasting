import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

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


def get_MA_EMA(df):
    ma_5 = df.rolling(window=5).mean()
    ma_10 = df.rolling(window=10).mean()
    ma_25 = df.rolling(window=25).mean()
    ma_50 = df.rolling(window=50).mean()
    ma_100 = df.rolling(window=100).mean()
    #ma_200 = df.iloc[:,0].rolling(window=200).mean()

    ema_20 = df.ewm(span = 20).mean()
    ema_30 = df.ewm(span = 30).mean()

    diff_p_5 = df - ma_5
    diff_p_10 = df - ma_10
    diff_p_25 = df - ma_25
    diff_P_100 = df - ma_100

    diff_p_ema_20 = df - ema_20
    diff_P_ema_30 = df -ema_30

    diff_25_50 = ma_25 - ma_50
    diff_25_100 = ma_25 - ma_100
    diff_50_100 = ma_50 - ma_100

    diff_ema_20_ema_30 = ema_20 - ema_30

    diff_ma_5_ema_20 = ma_5 - ema_20
    diff_ma_5_ema_30 = ma_5 - ema_30
    diff_ema_20_ma_50 = ema_20 - ma_50
    return diff_p_5, diff_p_10, diff_p_25, diff_P_100, diff_p_ema_20, diff_P_ema_30, diff_25_50, diff_25_100, diff_50_100, diff_ema_20_ema_30, diff_ma_5_ema_20, diff_ma_5_ema_30, diff_ema_20_ema_30




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


df_a = get_drawdown_ath()
df = get_btc_OHLC()
df['rsi_14']= get_rsi(df,14)
df_a[1],df_a[2],df_a[3],df_a[4],df_a[5],df_a[6], df_a[7], df_a[8], df_a[9], df_a[10], df_a[11], df_a[12], df_a[13] = get_MA_EMA(df_a)
df['SMI'] = get_SMI(df)
df['CCI'] = get_CCI(df)
df['william_a/d'] = get_william_A_D(df)
df['ATR'] = get_ATR(df)

#df.to_csv('../data/data.csv')

if __name__ == '__main__':
    print(df_a)
    df_a.to_csv('ma_ema.csv')

