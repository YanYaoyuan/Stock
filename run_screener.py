#!/usr/bin/env python3
"""
🦞 龙虾量化 — 选股入口
用法:
  python run_screener.py                                 # 默认沪深300, 纯基本面
  python run_screener.py --pool zz500                    # 中证500
  python run_screener.py --factor-preprocess             # 启用因子预处理 (去极值+中性化+Z-Score)
  python run_screener.py --sentiment                     # 启用新闻情绪过滤
  python run_screener.py --factor-preprocess --sentiment # 因子预处理 + 情绪
  python run_screener.py --market                        # 查看市场整体情绪
"""

import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ROE_MIN, PROFIT_GROWTH_MIN, PE_MAX, PB_MAX, OUTPUT_DIR,
    SENTIMENT_ENABLED, STOCK_SENTIMENT_MIN, MARKET_SENTIMENT_MIN,
    NEWS_CHECK_ENABLED, NEWS_NEGATIVE_THRESHOLD,
)
from src.data.datasource import BaostockDataSource
from src.screener.stock_screener import StockScreener


class ScreenConfig:
    ROE_MIN = ROE_MIN
    PROFIT_GROWTH_MIN = PROFIT_GROWTH_MIN
    SENTIMENT_ENABLED = SENTIMENT_ENABLED
    STOCK_SENTIMENT_MIN = STOCK_SENTIMENT_MIN
    MARKET_SENTIMENT_MIN = MARKET_SENTIMENT_MIN
    NEWS_CHECK_ENABLED = NEWS_CHECK_ENABLED
    NEWS_NEGATIVE_THRESHOLD = NEWS_NEGATIVE_THRESHOLD


def main():
    parser = argparse.ArgumentParser(description='🦞 龙虾选股')
    parser.add_argument('--pool', type=str, default='hs300', choices=['hs300', 'zz500'])
    parser.add_argument('--factor-preprocess', action='store_true',
                        help='启用因子预处理 (去极值→行业中性化→Z-Score)')
    parser.add_argument('--sentiment', action='store_true', help='启用新闻情绪过滤')
    parser.add_argument('--market', action='store_true', help='查看市场整体情绪')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    log = logging.getLogger(__name__)

    sentiment_engine = None
    if args.sentiment or args.market:
        try:
            from src.utils.sentiment import MarketSentimentEngine
            sentiment_engine = MarketSentimentEngine()
        except ImportError as e:
            print(f"❌ 情绪模块导入失败: {e}")
            return

    # 市场情绪查看
    if args.market:
        print("🦞 市场整体情绪\n" + "=" * 60)
        sentiment = sentiment_engine.get_market_sentiment()
        score = sentiment['score']
        bar = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "🟡")
        print(f"   情绪指数: {bar} {score:+.2f}")
        print(f"   新闻数量: {sentiment['news_count']} 条")
        print(f"   利好: {sentiment['positive_news']} 条 | 利空: {sentiment['negative_news']} 条")
        if sentiment['hot_topics']:
            print(f"\n   🔥 热门话题:")
            for topic in sentiment['hot_topics']:
                icon = "📈" if topic['sentiment'] == 'positive' else "📉"
                print(f"      {icon} {topic['topic'][:50]}")
        if not args.sentiment:
            return

    # 选股
    features = []
    if args.factor_preprocess:
        features.append("因子预处理")
    if args.sentiment:
        features.append("情绪")
    mode_str = " + ".join(features) if features else "纯基本面"

    print(f"\n🦞 龙虾选股 — {args.pool} ({mode_str})")
    print(f"   条件: ROE>{ROE_MIN*100:.0f}% 或 增长>{PROFIT_GROWTH_MIN*100:.0f}%")
    print(f"   估值: PE<{PE_MAX}, PB<{PB_MAX}")
    if args.factor_preprocess:
        print(f"   预处理: MAD去极值 → 行业中性化 → Z-Score")
        print(f"   加权: ROE×30% + 成长×30% + EP×20% + BP×20%")
    if args.sentiment:
        print(f"   情绪: 个股>{STOCK_SENTIMENT_MIN}, 市场>{MARKET_SENTIMENT_MIN}")
    print("=" * 60)

    with BaostockDataSource() as ds:
        pool = ds.get_stock_pool(args.pool)
        print(f"   股票池: {len(pool)} 只")

        # 获取行业分类 (因子预处理用)
        industry_map = None
        if args.factor_preprocess:
            from src.utils.factor_preprocess import get_industry_map_from_codes
            codes = [s['code'] for s in pool]
            industry_map = get_industry_map_from_codes(codes, ds)

        screener = StockScreener(ds, ScreenConfig())
        passed = screener.screen(
            pool,
            sentiment_engine=sentiment_engine if args.sentiment else None,
            use_factor_preprocess=args.factor_preprocess,
            industry_map=industry_map,
        )

        print(f"\n{'='*60}")
        print(f"🏆 选股通过: {len(passed)} 只")
        print(f"{'='*60}")

        if args.factor_preprocess:
            print(f"{'#':<4} {'代码':<12} {'名称':<10} {'ROE':<8} {'增长':<8} "
                  f"{'Z_ROE':<7} {'Z_增长':<7} {'因子分':<7}")
            print("-" * 70)
            for i, s in enumerate(passed):
                z_roe = f"{s['z_roe']:+.2f}" if s.get('z_roe') is not None else '—'
                z_g = f"{s['z_growth']:+.2f}" if s.get('z_growth') is not None else '—'
                fs = f"{s['factor_score']:+.3f}" if s.get('factor_score') is not None else '—'
                print(f"{i+1:<4} {s['code']:<12} {s['name']:<10} "
                      f"{s['roe'] or '—':>8} {s['growth'] or '—':>8} "
                      f"{z_roe:>7} {z_g:>7} {fs:>7}")
        elif args.sentiment:
            print(f"{'#':<4} {'代码':<12} {'名称':<10} {'ROE':<8} {'增长':<8} {'情绪':<8}")
            print("-" * 55)
            for i, s in enumerate(passed):
                sent_str = f"{s['sentiment_score']:+.2f}" if s.get('sentiment_score') is not None else '—'
                print(f"{i+1:<4} {s['code']:<12} {s['name']:<10} "
                      f"{s['roe'] or '—':>8} {s['growth'] or '—':>8} {sent_str:>8}")
        else:
            print(f"{'#':<4} {'代码':<12} {'名称':<10} {'ROE':<8} {'增长':<8}")
            print("-" * 45)
            for i, s in enumerate(sorted(passed, key=lambda x: (x['roe'] or 0), reverse=True)):
                print(f"{i+1:<4} {s['code']:<12} {s['name']:<10} "
                      f"{s['roe'] or '—':>8} {s['growth'] or '—':>8}")

        # 保存结果
        import pandas as pd
        out_dir = os.path.join(OUTPUT_DIR, 'reports')
        os.makedirs(out_dir, exist_ok=True)
        from datetime import datetime
        suffix_parts = []
        if args.factor_preprocess:
            suffix_parts.append('factor')
        if args.sentiment:
            suffix_parts.append('sentiment')
        suffix = '_' + '_'.join(suffix_parts) if suffix_parts else ''
        csv_path = os.path.join(out_dir, f"screener_{datetime.now().strftime('%Y%m%d')}{suffix}.csv")
        pd.DataFrame(passed).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n📄 结果: {csv_path}")

    # 生成可视化图表
    if passed:
        try:
            from src.utils.visualizer import generate_screener_chart
            chart_dir = os.path.join(OUTPUT_DIR, 'charts')
            img = generate_screener_chart(passed, chart_dir)
            if img:
                log.info(f"📊 选股可视化: {img}")
        except Exception as e:
            log.warning(f"可视化生成失败: {e}")


if __name__ == "__main__":
    main()
