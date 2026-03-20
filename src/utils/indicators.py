# 🦞 龙虾量化 — 技术指标计算工具

import numpy as np
import pandas as pd


def calc_ma(series, periods=(5, 20, 60)):
    """计算移动平均线"""
    result = {}
    for p in periods:
        result[f'MA{p}'] = series.rolling(p).mean()
    return result


def calc_macd(series, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = 2 * (dif - dea)
    return dif, dea, macd


def calc_rsi(series, period=14):
    """计算RSI指标"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    return rsi


def calc_all_indicators(df):
    """对DataFrame计算全部技术指标"""
    df = df.copy()
    
    # 均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # MACD
    df['DIF'], df['DEA'], df['MACD'] = calc_macd(df['close'])
    
    # RSI
    df['RSI'] = calc_rsi(df['close'])
    
    # 量比
    df['VOL_MA5'] = df['volume'].rolling(5).mean()
    
    return df


def is_bullish_alignment(row, spread_min=0.005):
    """判断多头排列: MA5 > MA20 > MA60"""
    ma5, ma20, ma60 = row['MA5'], row['MA20'], row['MA60']
    if pd.isna(ma5) or pd.isna(ma20) or pd.isna(ma60):
        return False
    if not (ma5 > ma20 > ma60):
        return False
    if (ma5 - ma20) / ma20 < spread_min:
        return False
    return True


def is_macd_bullish(row):
    """MACD看多: DIF > DEA"""
    if pd.isna(row['DIF']) or pd.isna(row['DEA']):
        return False
    return row['DIF'] > row['DEA']


def is_rsi_normal(row, low=30, high=65):
    """RSI在正常区间"""
    if pd.isna(row['RSI']):
        return False
    return low <= row['RSI'] <= high


def is_volume_surge(row, ratio_min=1.1):
    """成交量放大"""
    if pd.isna(row['volume']) or pd.isna(row['VOL_MA5']):
        return False
    if row['VOL_MA5'] <= 0:
        return False
    return row['volume'] / row['VOL_MA5'] >= ratio_min
