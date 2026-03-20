#!/usr/bin/env python3
"""
🦞 龙虾量化 — 实时情绪分析入口

用法:
  python run_sentiment.py                    # 市场整体情绪
  python run_sentiment.py --stock sh.600519  # 个股情绪
  python run_sentiment.py --stock sh.600519 --name 贵州茅台
  python run_sentiment.py --watchlist 沪深300  # 扫描自选股
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.sentiment import MarketSentimentEngine
from src.data.datasource import BaostockDataSource


def main():
    parser = argparse.ArgumentParser(description='🦞 龙虾实时情绪分析')
    parser.add_argument('--stock', type=str, default=None, help='股票代码 (如 sh.600519)')
    parser.add_argument('--name', type=str, default=None, help='股票名称')
    parser.add_argument('--market', action='store_true', help='查看市场整体情绪')
    parser.add_argument('--watchlist', type=str, default=None, choices=['hs300', 'zz500'],
                        help='扫描股票池情绪')
    args = parser.parse_args()
    
    engine = MarketSentimentEngine()
    
    # === 市场情绪 ===
    if args.market or (not args.stock and not args.watchlist):
        print("🦞 市场整体情绪扫描\n" + "=" * 60)
        sentiment = engine.get_market_sentiment()
        
        score = sentiment['score']
        bar = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "🟡")
        
        print(f"   情绪指数: {bar} {score:+.2f}")
        print(f"   新闻数量: {sentiment['news_count']} 条")
        print(f"   利好: {sentiment['positive_news']} 条 | 利空: {sentiment['negative_news']} 条")
        print(f"   更新时间: {sentiment['timestamp']}")
        
        if sentiment['hot_topics']:
            print(f"\n   🔥 热门话题:")
            for topic in sentiment['hot_topics']:
                icon = "📈" if topic['sentiment'] == 'positive' else "📉"
                print(f"      {icon} {topic['topic'][:50]}")
        
        # 交易建议
        print(f"\n   💡 建议: ", end="")
        if score > 0.3:
            print("市场情绪偏多，适合买入优质标的")
        elif score < -0.3:
            print("市场情绪偏空，建议降低仓位或观望")
        else:
            print("市场情绪中性，正常操作")
        
        return
    
    # === 个股情绪 ===
    if args.stock:
        print(f"🦞 个股情绪分析: {args.stock} {args.name or ''}\n" + "=" * 60)
        result = engine.get_stock_sentiment(args.stock, args.name)
        
        score = result['sentiment_score']
        bar = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "🟡")
        
        print(f"   股票: {result['name']} ({result['code']})")
        print(f"   综合情绪: {bar} {score:+.2f}")
        print(f"   新闻情绪: {result['news_sentiment']:+.2f} ({result['news_count']}条)")
        print(f"   行情情绪: {result['market_sentiment']:+.2f}")
        print(f"   更新时间: {result['timestamp']}")
        
        if result.get('realtime'):
            rt = result['realtime']
            print(f"\n   📊 实时行情:")
            print(f"      最新价: {rt['price']:.2f}")
            print(f"      涨跌幅: {rt['change_pct']:+.2f}%")
            print(f"      成交量: {rt['volume']:.0f}手")
        
        if result.get('signals'):
            print(f"\n   📰 关键信号: {', '.join(result['signals'][:8])}")
        
        print(f"\n   💡 建议: ", end="")
        if score > 0.3:
            print("情绪积极，可考虑买入")
        elif score < -0.3:
            print("情绪消极，建议回避")
        else:
            print("情绪中性")
        
        return
    
    # === 自选股扫描 ===
    if args.watchlist:
        print(f"🦞 扫描 {args.watchlist} 情绪\n" + "=" * 60)
        
        with BaostockDataSource() as ds:
            pool = ds.get_stock_pool(args.watchlist)
        
        print(f"   股票池: {len(pool)} 只，正在扫描...\n")
        results = engine.scan_watchlist(pool)
        
        # 按情绪分类
        positive = [r for r in results if r.get('sentiment_score', 0) > 0.2]
        negative = [r for r in results if r.get('sentiment_score', 0) < -0.2]
        neutral = [r for r in results if -0.2 <= r.get('sentiment_score', 0) <= 0.2]
        
        print(f"   🟢 情绪积极: {len(positive)} 只")
        print(f"   🟡 情绪中性: {len(neutral)} 只")
        print(f"   🔴 情绪消极: {len(negative)} 只")
        
        if positive:
            print(f"\n   📈 积极情绪 Top 15:")
            print(f"   {'#':<4} {'代码':<12} {'名称':<10} {'情绪':<8} {'信号'}")
            print(f"   {'-'*55}")
            for i, r in enumerate(positive[:15]):
                signals = ', '.join(r.get('signals', [])[:3]) or '-'
                print(f"   {i+1:<4} {r['code']:<12} {r['name']:<10} {r['sentiment_score']:+.2f}   {signals}")
        
        if negative:
            print(f"\n   📉 消极情绪 Top 10:")
            for i, r in enumerate(negative[:10]):
                print(f"   {i+1:<4} {r['code']:<12} {r['name']:<10} {r['sentiment_score']:+.2f}")
        
        return


if __name__ == "__main__":
    main()
