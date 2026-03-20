# 🦞 龙虾量化 — 历史情绪数据缓存 (用于回测)

"""
回测时无法实时抓取新闻，需要预先生成历史情绪数据文件。
本模块提供缓存加载和查询功能。

数据格式 (CSV):
  date,code,sentiment_score,news_sentiment,market_sentiment
  2024-01-15,sh.600519,0.45,0.50,0.20
  2024-01-15,sz.000858,0.12,0.10,0.20
  ...

生成方式:
  1. 实盘运行时定期保存情绪数据 (future: 自动化)
  2. 手动通过 run_sentiment.py 批量采集后整理
  3. generate_template() 生成空白模板

用法:
  cache = SentimentDataCache()
  cache.load_from_csv("sentiment_data.csv")
  s = cache.get_sentiment("sh.600519", "2024-01-15")
  # s = {'sentiment_score': 0.45, 'news_sentiment': 0.50, 'market_sentiment': 0.20} or None
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

log = logging.getLogger(__name__)


class SentimentDataCache:
    """历史情绪数据缓存 — 回测用"""

    def __init__(self):
        self._data = None  # DataFrame
        self._index = None  # MultiIndex dict: (code, date_str) -> row
        self._market = None  # dict: date_str -> market_sentiment
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def load_from_csv(self, file_path):
        """
        从CSV加载历史情绪数据

        Args:
            file_path: CSV文件路径

        Returns:
            True 加载成功, False 文件不存在或为空
        """
        if not os.path.exists(file_path):
            log.warning(f"情绪缓存文件不存在: {file_path}")
            return False

        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            if df.empty:
                log.warning(f"情绪缓存文件为空: {file_path}")
                return False

            # 标准化列名
            df.columns = [c.strip().lower() for c in df.columns]

            required = ['date', 'code']
            for col in required:
                if col not in df.columns:
                    log.error(f"情绪缓存缺少必要列: {col}")
                    return False

            # 标准化日期格式
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # 标准化代码格式: 统一小写
            df['code'] = df['code'].str.strip().str.lower()

            # 填充缺失的数值列为0
            for col in ['sentiment_score', 'news_sentiment', 'market_sentiment']:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

            self._data = df

            # 构建索引: {(code, date_str): row_dict}
            self._index = {}
            for _, row in df.iterrows():
                key = (row['code'], row['date'])
                self._index[key] = {
                    'sentiment_score': float(row['sentiment_score']),
                    'news_sentiment': float(row['news_sentiment']),
                    'market_sentiment': float(row['market_sentiment']),
                }

            # 构建市场情绪索引: {date_str: market_sentiment}
            self._market = {}
            for date, group in df.groupby('date'):
                self._market[date] = float(group['market_sentiment'].iloc[0])

            self._loaded = True
            log.info(f"情绪缓存加载成功: {len(df)} 条记录, "
                     f"{df['code'].nunique()} 只股票, "
                     f"{df['date'].nunique()} 个交易日")
            return True

        except Exception as e:
            log.error(f"情绪缓存加载失败: {e}")
            return False

    def get_sentiment(self, code, date):
        """
        获取指定股票在指定日期的情绪数据

        Args:
            code: 股票代码 (如 'sh.600519')
            date: 日期，可以是 str(YYYY-MM-DD) 或 datetime 或 pd.Timestamp

        Returns:
            dict {'sentiment_score': float, 'news_sentiment': float, 'market_sentiment': float}
            或 None (无数据时)
        """
        if not self._loaded or self._index is None:
            return None

        date_str = self._to_date_str(date)
        code_clean = str(code).strip().lower()

        return self._index.get((code_clean, date_str))

    def get_market_sentiment(self, date):
        """
        获取指定日期的市场整体情绪

        Args:
            date: 日期

        Returns:
            float 或 None
        """
        if not self._loaded or self._market is None:
            return None

        date_str = self._to_date_str(date)
        return self._market.get(date_str)

    def get_available_dates(self):
        """返回所有有数据的交易日列表"""
        if not self._loaded or self._data is None:
            return []
        return sorted(self._data['date'].unique().tolist())

    def get_available_codes(self, date=None):
        """
        返回有数据的股票代码列表
        date: 指定日期则返回该日有数据的股票，None则返回全部
        """
        if not self._loaded or self._data is None:
            return []
        if date is None:
            return sorted(self._data['code'].unique().tolist())
        date_str = self._to_date_str(date)
        subset = self._data[self._data['date'] == date_str]
        return sorted(subset['code'].unique().tolist())

    def stats(self):
        """返回缓存统计信息"""
        if not self._loaded:
            return "情绪缓存未加载"
        return (f"记录数: {len(self._data)}, "
                f"股票数: {self._data['code'].nunique()}, "
                f"交易日: {self._data['date'].nunique()}, "
                f"日期范围: {self._data['date'].min()} ~ {self._data['date'].max()}")

    @staticmethod
    def _to_date_str(date):
        """统一转换为 YYYY-MM-DD 字符串"""
        if isinstance(date, str):
            return date[:10]  # 防止带时间
        elif isinstance(date, (pd.Timestamp, datetime)):
            return date.strftime('%Y-%m-%d')
        else:
            return str(date)[:10]

    @staticmethod
    def generate_template(file_path="sentiment_template.csv", days=30):
        """
        生成空白模板CSV，方便手动填充

        Args:
            file_path: 输出路径
            days: 默认生成多少天的空行
        """
        from datetime import timedelta

        today = datetime.now()
        rows = []
        for i in range(days):
            d = (today - timedelta(days=days - 1 - i)).strftime('%Y-%m-%d')
            rows.append({
                'date': d,
                'code': 'sh.600519',  # 示例
                'sentiment_score': 0.0,
                'news_sentiment': 0.0,
                'market_sentiment': 0.0,
            })

        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        log.info(f"情绪模板已生成: {file_path} ({days}行)")
        return file_path
