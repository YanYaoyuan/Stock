# 🦞 龙虾量化 — 因子预处理模块

"""
标准量化因子预处理流水线:
  去极值 (Winsorization) → 行业+市值中性化 (Neutralization) → Z-Score标准化

为什么必须做:
  - 不做去极值: 一个异常值拉高均值, 导致大部分股票Z-Score接近0
  - 不做中性化: 策略变成行业/市值配置策略, 不是选股Alpha策略
    (比如直接按PE低到高选, 组合里全是银行股)
  - 不做Z-Score: 不同量纲因子无法横向加权 (PE是倍数, ROE是百分比)

MAD去极值 vs 3σ:
  MAD (Median Absolute Deviation) 在A股东尾分布下比3σ更鲁棒

中性化方法:
  OLS回归: Factor = β·ln(Size) + Σ β_i·Industry_i + ε
  取残差 ε 作为中性化后因子

Z-Score: z = (x - μ) / σ, 均值0标准差1

用法:
  preprocessor = FactorPreprocessor()
  # 横截面数据: DataFrame, index=code, columns=因子名
  df_neutralized = preprocessor.process(df, industry_map, market_cap_series)
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


class FactorPreprocessor:
    """因子预处理: 去极值 → 中性化 → 标准化"""

    def __init__(self, winsor_method='mad', mad_n=5, neutralize=True, zscore=True):
        """
        Args:
            winsor_method: 去极值方法, 'mad' (中位数绝对偏差) 或 'sigma' (3σ)
            mad_n: MAD方法的倍数, 默认5 (比3σ更宽松, 适合A股东尾)
            neutralize: 是否做行业+市值中性化
            zscore: 是否做Z-Score标准化
        """
        self.winsor_method = winsor_method
        self.mad_n = mad_n
        self.neutralize = neutralize
        self.zscore = zscore

    def process(self, df, industry_map=None, market_cap=None):
        """
        完整预处理流水线

        Args:
            df: DataFrame, index=股票代码, columns=因子列
                例: pd.DataFrame({'ROE': [15, 20, ...], 'PE': [10, 30, ...]}, index=['sh.600519', ...])
            industry_map: dict, {股票代码: 行业名称}
                例: {'sh.600519': '食品饮料', 'sh.601398': '银行'}
            market_cap: Series, index=股票代码, values=总市值
                例: pd.Series([1000, 5000, ...], index=['sh.600519', ...])

        Returns:
            DataFrame: 预处理后的因子值, 同shape
        """
        result = df.copy()

        # Step 1: 去极值
        for col in result.columns:
            result[col] = self.winsorize(result[col])

        # Step 2: 中性化
        if self.neutralize and industry_map is not None and market_cap is not None:
            for col in result.columns:
                result[col] = self.neutralize_factor(
                    result[col], industry_map, market_cap
                )

        # Step 3: Z-Score
        if self.zscore:
            for col in result.columns:
                result[col] = self.standardize(result[col])

        return result

    def winsorize(self, series):
        """
        去极值: 将超出边界的值拉回到边界

        MAD方法 (推荐):
          median = 中位数
          MAD = median(|x_i - median|)
          upper = median + n * 1.4826 * MAD
          lower = median - n * 1.4826 * MAD
          1.4826是正态分布下MAD到标准差的转换系数

        3σ方法:
          upper = mean + 3 * std
          lower = mean - 3 * std
        """
        s = series.dropna()
        if len(s) == 0:
            return series

        if self.winsor_method == 'mad':
            median = s.median()
            mad = np.median(np.abs(s - median))
            if mad == 0 or np.isnan(mad):
                return series  # 所有值相同, 无法去极值
            upper = median + self.mad_n * 1.4826 * mad
            lower = median - self.mad_n * 1.4826 * mad
        else:
            mean = s.mean()
            std = s.std()
            if std == 0 or np.isnan(std):
                return series
            upper = mean + 3 * std
            lower = mean - 3 * std

        return series.clip(lower=lower, upper=upper)

    def neutralize_factor(self, factor, industry_map, market_cap):
        """
        行业 + 市值中性化

        方法: OLS回归取残差
          Y = 去极值后的因子
          X = ln(市值) + 行业哑变量

        Args:
            factor: Series, index=股票代码
            industry_map: dict, {代码: 行业}
            market_cap: Series, index=股票代码, values=市值(元)

        Returns:
            Series: 中性化后的因子 (残差)
        """
        # 对齐数据
        common_idx = factor.dropna().index.intersection(market_cap.dropna().index)
        codes_with_industry = [c for c in common_idx if c in industry_map]

        if len(codes_with_industry) < 10:
            log.debug(f"中性化数据不足({len(codes_with_industry)}只), 跳过")
            return factor

        # 构建X矩阵
        y = factor.loc[codes_with_industry].values

        # 市值对数
        ln_mcap = np.log(market_cap.loc[codes_with_industry].clip(lower=1).values)

        # 行业哑变量
        industries = [industry_map[c] for c in codes_with_industry]
        unique_industries = sorted(set(industries))

        # 去掉最后一个行业 (避免共线性)
        n_industries = len(unique_industries)
        if n_industries <= 1:
            log.debug(f"行业数不足({n_industries}), 仅做市值中性化")
            # 仅市值中性化
            X = np.column_stack([np.ones(len(codes_with_industry)), ln_mcap])
        else:
            # 行业哑变量矩阵 (去掉最后一个)
            dummy = np.zeros((len(codes_with_industry), n_industries - 1))
            for i, ind in enumerate(industries):
                if ind in unique_industries[:-1]:
                    dummy[i, unique_industries[:-1].index(ind)] = 1

            X = np.column_stack([np.ones(len(codes_with_industry)), ln_mcap, dummy])

        # OLS回归
        try:
            # 使用最小二乘法: β = (X'X)^(-1) X'y
            XtX = X.T @ X
            Xty = X.T @ y

            # 正则化防止奇异矩阵
            XtX += np.eye(XtX.shape[0]) * 1e-8

            beta = np.linalg.solve(XtX, Xty)
            fitted = X @ beta
            residuals = y - fitted
        except np.linalg.LinAlgError:
            log.warning("中性化回归失败 (矩阵奇异), 返回原始因子")
            return factor

        # 构建结果Series
        result = factor.copy()
        result.loc[codes_with_industry] = residuals

        # 没有行业或市值数据的保持NaN
        return result

    def standardize(self, series):
        """
        Z-Score标准化: z = (x - μ) / σ

        结果: 均值=0, 标准差=1
        """
        s = series.dropna()
        if len(s) == 0:
            return series

        mean = s.mean()
        std = s.std()

        if std == 0 or np.isnan(std):
            log.debug("标准差为0, 无法标准化")
            return series * 0  # 所有值相同, 返回0

        result = series.copy().astype(float)
        result.loc[s.index] = ((s - mean) / std).values
        return result

    @staticmethod
    def rank_normalize(series):
        """
        排序标准化 (Rank): 将因子值转为分位数 0~1

        优点: 完全消除分布形状影响, 对离群值极度鲁棒
        适用: 因子分布极度非正态时替代Z-Score
        """
        s = series.dropna()
        if len(s) == 0:
            return series

        result = series.copy()
        result.loc[s.index] = s.rank(pct=True)
        return result


# ============================================================
#  辅助函数
# ============================================================

def get_industry_map_from_codes(codes, datasource=None):
    """
    获取股票的行业分类映射

    如果提供了datasource (BaostockDataSource), 尝试从baostock获取
    否则返回空dict

    Args:
        codes: 股票代码列表
        datasource: BaostockDataSource 实例 (可选)

    Returns:
        dict: {代码: 行业名称}
    """
    if datasource is None:
        return {}

    try:
        import baostock as bs
        industry_map = {}
        for code in codes:
            try:
                rs = bs.query_stock_industry(code=code)
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rows and len(rows[0]) > 3:
                    industry_map[code] = rows[0][3]  # industry_classification
                else:
                    # 回退: 用baostock的stock_basic获取
                    rs2 = bs.query_stock_basic(code=code)
                    rows2 = []
                    while rs2.next():
                        rows2.append(rs2.get_row_data())
                    if rows2 and len(rows2[0]) > 1:
                        industry_map[code] = rows2[0][1]  # industry
            except Exception:
                continue

        if industry_map:
            log.info(f"   行业分类获取成功: {len(industry_map)}/{len(codes)} 只")
        else:
            log.info("   行业分类未获取到 (将跳过行业中性化)")
        return industry_map

    except Exception as e:
        log.warning(f"行业分类获取失败: {e}")
        return {}


def compute_market_cap(kline_df):
    """
    从K线数据计算总市值 (近似)
    baostock不直接提供市值, 用 收盘价 × 总股本 近似
    如果没有总股本数据, 用close作为替代排名依据

    Args:
        kline_df: DataFrame with 'close' column

    Returns:
        float: 最近一个交易日的收盘价 (作为市值代理)
    """
    if kline_df is None or kline_df.empty:
        return np.nan
    return kline_df['close'].iloc[-1]
