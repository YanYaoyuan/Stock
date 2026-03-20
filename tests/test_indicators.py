# 🦞 龙虾量化 — 技术指标测试

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.utils.indicators import calc_all_indicators, is_bullish_alignment, is_macd_bullish


def test_calc_indicators():
    """测试技术指标计算"""
    np.random.seed(42)
    n = 100
    prices = 10 + np.cumsum(np.random.randn(n) * 0.5)
    volumes = np.random.randint(100000, 500000, n)
    
    df = pd.DataFrame({
        'close': prices,
        'volume': volumes,
    })
    
    result = calc_all_indicators(df)
    
    # 检查列存在
    for col in ['MA5', 'MA20', 'MA60', 'DIF', 'DEA', 'MACD', 'RSI', 'VOL_MA5']:
        assert col in result.columns, f"缺少列: {col}"
    
    # MA5前4行应该是NaN
    assert result['MA5'][:4].isna().all()
    # MA60前59行应该是NaN
    assert result['MA60'][:59].isna().all()
    
    # RSI范围应该在0-100之间
    valid_rsi = result['RSI'].dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
    
    print("✅ test_calc_indicators passed")


def test_bullish_alignment():
    """测试多头排列判断"""
    row = {'MA5': 11.0, 'MA20': 10.5, 'MA60': 10.0}
    assert is_bullish_alignment(row) == True
    
    row2 = {'MA5': 10.0, 'MA20': 10.5, 'MA60': 10.0}
    assert is_bullish_alignment(row2) == False
    
    print("✅ test_bullish_alignment passed")


def test_macd():
    """测试MACD判断"""
    row = {'DIF': 0.5, 'DEA': 0.3}
    assert is_macd_bullish(row) == True
    
    row2 = {'DIF': 0.2, 'DEA': 0.5}
    assert is_macd_bullish(row2) == False
    
    print("✅ test_macd passed")


if __name__ == "__main__":
    test_calc_indicators()
    test_bullish_alignment()
    test_macd()
    print("\n✅ All tests passed!")
