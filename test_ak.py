import akshare as ak
import pandas as pd

ticker = "600519"
prefix = "sh" if ticker.startswith("6") else "sz"
prefixed = f"{prefix}{ticker}"
df = ak.stock_zh_a_daily(symbol=prefixed, start_date="20200101", end_date="20240101", adjust="qfq")

df.rename(columns={
    'date': '日期',
    'open': '开盘',
    'high': '最高',
    'low': '最低',
    'close': '收盘',
    'volume': '成交量',
    'amount': '成交额',
    'turnover': '换手率'
}, inplace=True)

df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
df['前收盘'] = df['收盘'].shift(1)
df['涨跌额'] = df['收盘'] - df['前收盘']
df['涨跌幅'] = df['涨跌额'] / df['前收盘'] * 100
df['振幅'] = (df['最高'] - df['最低']) / df['前收盘'] * 100
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(df.head())
print(df.columns)
