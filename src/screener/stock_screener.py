# 🦞 龙虾量化 — 基本面选股引擎

import time
import logging
from datetime import datetime

log = logging.getLogger(__name__)


class StockScreener:
    """基本面选股: ROE + 利润增长 + 估值过滤 + 可选情绪过滤"""

    def __init__(self, datasource, config):
        self.ds = datasource
        self.cfg = config

    def screen(self, pool, ref_date=None, sentiment_engine=None):
        """
        对股票池进行基本面筛选

        Args:
            pool: [{'code': 'sh.600519', 'name': '贵州茅台'}, ...]
            ref_date: 参考日期 (用于确定财报期)
            sentiment_engine: 可选, MarketSentimentEngine实例, 传入则启用情绪过滤

        Returns:
            通过筛选的股票列表: [{'code': ..., 'name': ..., 'roe': ..., 'growth': ..., 'sentiment_score': ...}, ...]
        """
        if ref_date is None:
            ref_date = datetime.now()

        # 确定最新可用财报期
        fy = ref_date.year
        fq = (ref_date.month - 1) // 3  # 上一个季度
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

            # ROE或增长满足其一即可
            roe_ok = roe is not None and roe > self.cfg.ROE_MIN * 100
            growth_ok = growth is not None and growth > self.cfg.PROFIT_GROWTH_MIN * 100

            if roe_ok or growth_ok:
                item = {
                    'code': code,
                    'name': stock['name'],
                    'roe': roe,
                    'growth': growth,
                    'sentiment_score': None,
                }
                passed.append(item)

        log.info(f"   基本面通过: {len(passed)}/{len(pool)} 只")

        # === 情绪过滤 (可选) ===
        if sentiment_engine is not None:
            sentiment_enabled = getattr(self.cfg, 'SENTIMENT_ENABLED', True)
            if sentiment_enabled:
                passed = self._filter_by_sentiment(passed, sentiment_engine)
            else:
                log.info("   情绪因子已禁用 (SENTIMENT_ENABLED=False)")

        return passed

    def _filter_by_sentiment(self, stocks, engine):
        """
        用实时情绪过滤候选股

        Args:
            stocks: 基本面通过的股票列表
            engine: MarketSentimentEngine 实例

        Returns:
            过滤后的股票列表 (含 sentiment_score 字段)
        """
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
                           f"建议降低仓位或观望, 不做硬过滤")
        except Exception as e:
            log.warning(f"   市场情绪获取失败: {e}")
            market_score = 0

        # 逐只扫描个股情绪
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

                # 负面新闻数检查
                if news_check_enabled:
                    neg_count = sum(1 for d in result.get('details', [])
                                   if d.get('score', 0) < -0.2)
                    if neg_count >= news_neg_threshold:
                        log.info(f"   ❌ {stock['name']}({stock['code']}) "
                               f"负面新闻过多({neg_count}条), 跳过")
                        sentiment_blocked += 1
                        continue

                # 情绪分阈值检查
                if score < stock_sentiment_min:
                    log.debug(f"   跳过 {stock['name']}: 情绪{score:.2f} < {stock_sentiment_min}")
                    sentiment_blocked += 1
                    continue

                sentiment_ok += 1
                filtered.append(stock)

            except Exception as e:
                log.warning(f"   情绪扫描失败 {stock['code']}: {e}")
                stock['sentiment_score'] = 0  # 扫描失败不跳过，给中性分
                sentiment_errors += 1
                filtered.append(stock)

            # 避免请求过快
            if i < len(stocks) - 1:
                time.sleep(0.3)

        log.info(f"   情绪过滤: 通过{sentiment_ok} | 拦截{sentiment_blocked} | 异常{sentiment_errors}")

        # 按情绪分降序排列
        filtered.sort(key=lambda x: x.get('sentiment_score', 0) or 0, reverse=True)

        return filtered
