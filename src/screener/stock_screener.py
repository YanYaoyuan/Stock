# 🦞 龙虾量化 — 基本面选股引擎 (含因子预处理)

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime

log = logging.getLogger(__name__)


class StockScreener:
    """基本面选股: 因子预处理 → ROE + 利润增长 + 可选情绪过滤"""

    def __init__(self, datasource, config):
        self.ds = datasource
        self.cfg = config

    def screen(self, pool, ref_date=None, sentiment_engine=None,
               use_factor_preprocess=False, industry_map=None):
        """
        对股票池进行基本面筛选

        Args:
            pool: [{'code': 'sh.600519', 'name': '贵州茅台'}, ...]
            ref_date: 参考日期
            sentiment_engine: 可选, MarketSentimentEngine实例
            use_factor_preprocess: 是否启用因子预处理 (去极值+中性化+Z-Score)
            industry_map: dict, {代码: 行业} (中性化需要)

        Returns:
            通过筛选的股票列表
        """
        if ref_date is None:
            ref_date = datetime.now()

        # 确定最新可用财报期
        fy = ref_date.year
        fq = (ref_date.month - 1) // 3
        if fq == 0:
            fy -= 1
            fq = 4

        passed = []

        for i, stock in enumerate(pool):
            if (i + 1) % 100 == 0:
                log.info(f"   选股进度: {i+1}/{len(pool)} (通过:{len(passed)})")

            code = stock['code']

            # 查基本面
            fund = self.ds.get_fundamental(code, fy, fq)

            # 最新季报无数据则往前推
            if fund['roe'] is None and fund['profit_growth'] is None:
                if fq > 1:
                    fund2 = self.ds.get_fundamental(code, fy, fq - 1)
                    if fund2['roe'] is not None or fund2['profit_growth'] is not None:
                        fund = fund2
                else:
                    fund3 = self.ds.get_fundamental(code, fy - 1, 4)
                    if fund3['roe'] is not None or fund3['profit_growth'] is not None:
                        fund = fund3

            roe = fund.get('roe')
            growth = fund.get('profit_growth')

            item = {
                'code': code,
                'name': stock['name'],
                'roe': roe,
                'growth': growth,
                'sentiment_score': None,
                'factor_score': None,
                'z_roe': None,
                'z_growth': None,
                'z_pe': None,
                'z_pb': None,
                'pe': None,
                'pb': None,
            }

            # 粗筛: ROE或增长满足其一 (保留更多候选给因子打分)
            roe_ok = roe is not None and roe > self.cfg.ROE_MIN * 100
            growth_ok = growth is not None and growth > self.cfg.PROFIT_GROWTH_MIN * 100

            if roe_ok or growth_ok:
                # 获取最新PE/PB (用于因子预处理)
                try:
                    recent_date = (ref_date - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
                    today_str = ref_date.strftime('%Y-%m-%d')
                    kline = self.ds.get_kline(code, recent_date, today_str)
                    if kline is not None and not kline.empty:
                        last = kline.iloc[-1]
                        item['pe'] = last.get('peTTM')
                        item['pb'] = last.get('pbMRQ')
                except Exception:
                    pass

                passed.append(item)

        log.info(f"   基本面通过: {len(passed)}/{len(pool)} 只")

        # === 因子预处理 (可选) ===
        if use_factor_preprocess and len(passed) > 10:
            passed = self._factor_score(passed, industry_map)
        else:
            log.info("   因子预处理: 未启用")

        # === 情绪过滤 (可选) ===
        if sentiment_engine is not None:
            sentiment_enabled = getattr(self.cfg, 'SENTIMENT_ENABLED', True)
            if sentiment_enabled:
                passed = self._filter_by_sentiment(passed, sentiment_engine)
            else:
                log.info("   情绪因子已禁用 (SENTIMENT_ENABLED=False)")

        return passed

    def _factor_score(self, stocks, industry_map):
        """
        因子预处理 + 综合打分

        因子:
          - ROE (质量因子, 越高越好)
          - profit_growth (成长因子, 越高越好)
          - EP = 1/PE (价值因子, 用倒数因为PE越低越好 → EP越高越好)
          - BP = 1/PB (价值因子)

        流程:
          1. 构建横截面DataFrame
          2. 去极值 (MAD)
          3. 行业+市值中性化 (如果有行业数据)
          4. Z-Score标准化
          5. 加权打分
        """
        from ..utils.factor_preprocess import FactorPreprocessor

        log.info(f"   📐 因子预处理开始 ({len(stocks)}只)...")

        # 构建横截面因子DataFrame
        codes = [s['code'] for s in stocks]
        factor_data = {}

        # ROE
        roe_series = pd.Series(
            [s['roe'] for s in stocks],
            index=codes, dtype=float
        )
        factor_data['ROE'] = roe_series

        # 利润增长
        growth_series = pd.Series(
            [s['growth'] for s in stocks],
            index=codes, dtype=float
        )
        factor_data['GROWTH'] = growth_series

        # EP = 1/PE (价值因子)
        pe_series = pd.Series(
            [s.get('pe') for s in stocks],
            index=codes, dtype=float
        )
        # PE必须为正
        pe_series = pe_series[pe_series > 0]
        ep_series = 1.0 / pe_series
        factor_data['EP'] = ep_series.reindex(codes)

        # BP = 1/PB (价值因子)
        pb_series = pd.Series(
            [s.get('pb') for s in stocks],
            index=codes, dtype=float
        )
        pb_series = pb_series[pb_series > 0]
        bp_series = 1.0 / pb_series
        factor_data['BP'] = bp_series.reindex(codes)

        df = pd.DataFrame(factor_data)

        # 市值代理 (用PE或PB为正的股票数量衡量, 简单用PE本身作为代理)
        # 更准确的做法是获取真实市值, 这里用PE*PB的倒数近似
        mcap_proxy = pd.Series(
            [max(s.get('pe') or 999, 1) * max(s.get('pb') or 999, 1) for s in stocks],
            index=codes, dtype=float
        )

        # 预处理
        has_industry = industry_map is not None and len(industry_map) > 5

        preprocessor = FactorPreprocessor(
            winsor_method='mad',
            mad_n=5,
            neutralize=has_industry,
            zscore=True,
        )

        df_processed = preprocessor.process(
            df,
            industry_map=industry_map if has_industry else None,
            market_cap=mcap_proxy,
        )

        # 记录Z-Score到每个股票
        for s in stocks:
            code = s['code']
            if code in df_processed.index:
                s['z_roe'] = df_processed.loc[code, 'ROE'] if 'ROE' in df_processed.columns else None
                s['z_growth'] = df_processed.loc[code, 'GROWTH'] if 'GROWTH' in df_processed.columns else None
                s['z_pe'] = df_processed.loc[code, 'EP'] if 'EP' in df_processed.columns else None
                s['z_pb'] = df_processed.loc[code, 'BP'] if 'BP' in df_processed.columns else None

                # 综合因子得分 (加权: ROE 30% + 成长 30% + EP 20% + BP 20%)
                weights = {'ROE': 0.30, 'GROWTH': 0.30, 'EP': 0.20, 'BP': 0.20}
                score = 0
                w_total = 0
                for factor_name, weight in weights.items():
                    val = df_processed.loc[code, factor_name]
                    if not pd.isna(val):
                        score += val * weight
                        w_total += weight
                s['factor_score'] = score / w_total if w_total > 0 else 0

        # 按因子得分降序排列
        stocks.sort(key=lambda x: x.get('factor_score') or 0, reverse=True)

        valid_scores = [s['factor_score'] for s in stocks if s.get('factor_score') is not None]
        if valid_scores:
            avg_score = np.mean(valid_scores)
            log.info(f"   📐 因子预处理完成, 平均得分: {avg_score:.3f}")
            top3_names = []
            for s in stocks[:3]:
                n = s['name']
                sc = s['factor_score']
                top3_names.append(f"{n}({sc:.2f})")
            log.info(f"      Top3: {', '.join(top3_names)}")

        return stocks

    def _filter_by_sentiment(self, stocks, engine):
        """用实时情绪过滤候选股"""
        stock_sentiment_min = getattr(self.cfg, 'STOCK_SENTIMENT_MIN', -0.3)
        news_check_enabled = getattr(self.cfg, 'NEWS_CHECK_ENABLED', True)
        news_neg_threshold = getattr(self.cfg, 'NEWS_NEGATIVE_THRESHOLD', 3)

        # 先看市场整体情绪
        try:
            market = engine.get_market_sentiment()
            market_score = market.get('score', 0)
            log.info(f"   市场情绪: {market_score:+.2f} "
                     f"(利好:{market.get('positive_news', 0)} 利空:{market.get('negative_news', 0)})")

            market_min = getattr(self.cfg, 'MARKET_SENTIMENT_MIN', -0.2)
            if market_score < market_min:
                log.warning(f"   ⚠️ 市场情绪过低({market_score:.2f} < {market_min:.2f}), "
                           f"建议降低仓位或观望")
        except Exception as e:
            log.warning(f"   市场情绪获取失败: {e}")

        filtered = []
        sentiment_ok = 0
        sentiment_blocked = 0
        sentiment_errors = 0

        for i, stock in enumerate(stocks):
            if (i + 1) % 20 == 0:
                log.info(f"   情绪扫描: {i+1}/{len(stocks)}")

            try:
                result = engine.get_stock_sentiment(stock['code'], stock['name'])
                score = result.get('sentiment_score', 0)
                stock['sentiment_score'] = score

                if news_check_enabled:
                    neg_count = sum(1 for d in result.get('details', [])
                                   if d.get('score', 0) < -0.2)
                    if neg_count >= news_neg_threshold:
                        log.info(f"   ❌ {stock['name']}({stock['code']}) "
                               f"负面新闻过多({neg_count}条), 跳过")
                        sentiment_blocked += 1
                        continue

                if score < stock_sentiment_min:
                    sentiment_blocked += 1
                    continue

                sentiment_ok += 1
                filtered.append(stock)

            except Exception as e:
                log.warning(f"   情绪扫描失败 {stock['code']}: {e}")
                stock['sentiment_score'] = 0
                sentiment_errors += 1
                filtered.append(stock)

            if i < len(stocks) - 1:
                time.sleep(0.3)

        log.info(f"   情绪过滤: 通过{sentiment_ok} | 拦截{sentiment_blocked} | 异常{sentiment_errors}")
        filtered.sort(key=lambda x: x.get('sentiment_score', 0) or 0, reverse=True)
        return filtered
