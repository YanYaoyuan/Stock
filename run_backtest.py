#!/usr/bin/env python3
"""
🦞 龙虾量化 — 回测入口 (四因子: 基本面+技术面+情绪+因子预处理)
用法:
  python run_backtest.py                                                    # 默认沪深300, 2020-2025
  python run_backtest.py --pool zz500 --start 2022-01-01                   # 自定义参数
  python run_backtest.py --factor-preprocess                                # 启用因子预处理
  python run_backtest.py --sentiment --sentiment-cache data/sentiment.csv  # 开启情绪因子
  python run_backtest.py --factor-preprocess --sentiment --sentiment-cache data/sentiment.csv  # 全开
"""

import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ROE_MIN, PROFIT_GROWTH_MIN, PE_MAX, PB_MAX,
    STOP_LOSS, TAKE_PROFIT, TRAILING_STOP, TRAILING_ACTIVATE,
    MAX_HOLD_DAYS, MAX_POSITIONS, POSITION_PCT,
    COMMISSION, STAMP_TAX, SLIPPAGE, OUTPUT_DIR,
    SENTIMENT_ENABLED, MARKET_SENTIMENT_MIN, STOCK_SENTIMENT_MIN,
    STOCK_SENTIMENT_BOOST, NEWS_CHECK_ENABLED, NEWS_NEGATIVE_THRESHOLD,
)
from src.data.datasource import BaostockDataSource
from src.screener.stock_screener import StockScreener
from src.backtest.engine import BacktestEngine, generate_report
from src.utils.indicators import calc_all_indicators


class Config:
    ROE_MIN = ROE_MIN
    PROFIT_GROWTH_MIN = PROFIT_GROWTH_MIN
    PE_MAX = PE_MAX
    PB_MAX = PB_MAX
    RSI_LOW = 30
    RSI_HIGH = 65
    MA_SPREAD_MIN = 0.005
    VOL_RATIO_MIN = 1.1
    STOP_LOSS = STOP_LOSS
    TAKE_PROFIT = TAKE_PROFIT
    TRAILING_STOP = TRAILING_STOP
    TRAILING_ACTIVATE = TRAILING_ACTIVATE
    MAX_HOLD_DAYS = MAX_HOLD_DAYS
    MAX_POSITIONS = MAX_POSITIONS
    POSITION_PCT = POSITION_PCT
    COMMISSION = COMMISSION
    STAMP_TAX = STAMP_TAX
    SLIPPAGE = SLIPPAGE
    # 情绪参数
    SENTIMENT_ENABLED = SENTIMENT_ENABLED
    MARKET_SENTIMENT_MIN = MARKET_SENTIMENT_MIN
    STOCK_SENTIMENT_MIN = STOCK_SENTIMENT_MIN
    STOCK_SENTIMENT_BOOST = STOCK_SENTIMENT_BOOST
    NEWS_CHECK_ENABLED = NEWS_CHECK_ENABLED
    NEWS_NEGATIVE_THRESHOLD = NEWS_NEGATIVE_THRESHOLD


def main():
    parser = argparse.ArgumentParser(description='🦞 龙虾回测系统')
    parser.add_argument('--pool', type=str, default='hs300', choices=['hs300', 'zz500'])
    parser.add_argument('--start', type=str, default='2020-01-01')
    parser.add_argument('--end', type=str, default='2025-12-31')
    parser.add_argument('--cash', type=float, default=100000)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--factor-preprocess', action='store_true',
                        help='启用因子预处理 (去极值→行业中性化→Z-Score)')
    parser.add_argument('--factor-min-score', type=float, default=0.0,
                        help='因子得分最低门槛 (默认0.0, 取前50%%)')
    parser.add_argument('--factor-top-pct', type=float, default=0.5,
                        help='取因子分Top N%%的股票送入回测 (默认50%%)')
    parser.add_argument('--sentiment', action='store_true', help='启用情绪因子')
    parser.add_argument('--sentiment-cache', type=str, default=None,
                        help='历史情绪数据文件路径 (CSV格式)')
    args = parser.parse_args()

    output_dir = args.output or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                                mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    log = logging.getLogger(__name__)

    # 情绪缓存处理
    sentiment_cache = None
    sentiment_mode = "off"

    if args.sentiment:
        if args.sentiment_cache:
            try:
                from src.utils.sentiment_cache import SentimentDataCache
                sentiment_cache = SentimentDataCache()
                loaded = sentiment_cache.load_from_csv(args.sentiment_cache)
                if loaded:
                    sentiment_mode = f"cache:{args.sentiment_cache}"
                else:
                    log.warning("⚠️ 情绪缓存文件加载失败，情绪因子已自动禁用")
                    sentiment_cache = None
            except Exception as e:
                log.error(f"情绪缓存加载异常: {e}")
                sentiment_cache = None
        else:
            log.warning("⚠️ 回测模式需提供 --sentiment-cache 文件来加载历史情绪数据")
            log.warning("   情绪因子已自动禁用 (回测无法实时抓取新闻)")
            sentiment_mode = "off (no cache)"

    # 打印策略信息
    features = []
    features.append(f"ROE>{ROE_MIN*100:.0f}%或增长>{PROFIT_GROWTH_MIN*100:.0f}%")
    if args.factor_preprocess:
        features.append(f"因子预处理(Top{args.factor_top_pct*100:.0f}%)")
    features.append("MA多头+MACD+RSI")
    if sentiment_cache:
        features.append("情绪")
    feature_str = " | ".join(features)

    log.info("=" * 70)
    log.info("🦞 龙虾 A股量化交易系统 — 四因子策略")
    log.info("=" * 70)
    log.info(f"   因子链: {feature_str}")
    log.info(f"   风控: 止损{STOP_LOSS*100}% | 止盈{TAKE_PROFIT*100}% | 移动止损{TRAILING_STOP*100}% | 时间{MAX_HOLD_DAYS}天")
    log.info(f"   区间: {args.start} ~ {args.end} | 资金: ¥{args.cash:,.0f}")
    log.info("=" * 70)

    cfg = Config()

    with BaostockDataSource() as ds:
        # Step 1: 获取股票池
        pool = ds.get_stock_pool(args.pool)
        log.info(f"   股票池: {args.pool} ({len(pool)}只)")

        # Step 2: 获取行业分类 (因子预处理用)
        industry_map = None
        if args.factor_preprocess:
            from src.utils.factor_preprocess import get_industry_map_from_codes
            codes = [s['code'] for s in pool]
            industry_map = get_industry_map_from_codes(codes, ds)

        # Step 3: 基本面选股 + 因子预处理
        screener = StockScreener(ds, cfg)
        screened = screener.screen(
            pool,
            use_factor_preprocess=args.factor_preprocess,
            industry_map=industry_map,
        )

        if not screened:
            log.error("❌ 选股通过为0，无法回测")
            return

        # Step 4: 因子分筛选 (取Top N%)
        if args.factor_preprocess:
            before_count = len(screened)
            # 按因子分降序排列 (已在screener中排好)
            top_n = max(10, int(len(screened) * args.factor_top_pct))
            # 过滤掉因子分低于阈值的
            filtered = [s for s in screened if (s.get('factor_score') is not None
                        and s['factor_score'] >= args.factor_min_score)]
            filtered = filtered[:top_n]

            log.info(f"   因子筛选: {before_count} → {len(filtered)} "
                     f"(Top{args.factor_top_pct*100:.0f}%, 分≥{args.factor_min_score})")

            if filtered:
                log.info(f"      Top5: " + ", ".join(
                    f"{s['name']}({s['factor_score']:.2f})" for s in filtered[:5]))
            screened = filtered

            if not screened:
                log.error("❌ 因子筛选后为0，无法回测")
                return

        # Step 5: 下载K线 + 计算指标
        log.info(f"📈 下载K线数据...")
        stock_data = []
        for i, s in enumerate(screened):
            if (i + 1) % 50 == 0:
                log.info(f"   K线进度: {i+1}/{len(screened)}")
            try:
                df = ds.get_kline(s['code'], args.start, args.end)
                if df is not None and len(df) >= 65:
                    df = calc_all_indicators(df)
                    stock_data.append({
                        'code': s['code'], 'name': s['name'],
                        'roe': s['roe'], 'growth': s['growth'],
                        'factor_score': s.get('factor_score'),
                        'df': df,
                    })
            except Exception as e:
                log.warning(f"   K线失败 {s['code']}: {e}")

        log.info(f"   K线加载: {len(stock_data)} 只")

        if not stock_data:
            log.error("❌ 无K线数据")
            return

        # Step 6: 回测
        engine = BacktestEngine(cfg, sentiment_cache=sentiment_cache)
        trades, daily_values, stats = engine.run(stock_data, args.start, args.end, args.cash)

    # 打印结果
    print(f"\n{'='*70}")
    print(f"🦞 回测结果 — {args.pool}")
    if args.factor_preprocess:
        print(f"   📐 因子预处理: ✅ (Top{args.factor_top_pct*100:.0f}%)")
    if sentiment_cache:
        print(f"   📰 情绪因子: ✅")
    print(f"{'='*70}")
    print(f"   初始资金: ¥{args.cash:,.0f}")
    print(f"   最终权益: ¥{stats['final_value']:,.0f}")
    print(f"   总收益率: {stats['total_return']:+.2f}%")
    print(f"   年化收益: {stats['annual_return']:+.2f}%")
    print(f"   Sharpe: {stats['sharpe']:.2f}")
    print(f"   最大回撤: {stats['max_drawdown']:.2f}%")
    print(f"   交易: {stats['total_trades']}笔 (盈{stats['wins']} 亏{stats['losses']})")
    print(f"   胜率: {stats['win_rate']:.1f}%")
    print(f"   盈亏比: {stats['profit_factor']:.2f}")
    print(f"   平均持仓: {stats['avg_hold']:.1f}天")
    print(f"{'='*70}")

    if trades:
        print(f"\n📋 交易记录 (前30笔):")
        print(f"{'#':<4} {'股票':<10} {'买入':<8} {'卖出':<8} {'天':<4} {'盈亏':<10} {'%':<8} {'原因'}")
        print("-" * 65)
        for i, t in enumerate(trades[:30]):
            print(f"{i+1:<4} {t['name']:<10} {t['entry_price']:<8.2f} {t['exit_price']:<8.2f} "
                  f"{t['hold_days']:<4} {t['pnl']:<+10.0f} {t['pnl_pct']:<+7.1f}% {t['reason']}")
        if len(trades) > 30:
            print(f"   ... 还有 {len(trades)-30} 笔")

    # 生成报告
    rpt = generate_report(trades, daily_values, stats, args.pool, args.start, args.end, args.cash, output_dir)
    log.info(f"\n📄 报告: {rpt}")

    # 生成可视化图表
    try:
        from src.utils.visualizer import generate_backtest_charts
        chart_dir = os.path.join(output_dir, 'charts')
        images = generate_backtest_charts(
            trades, daily_values, stats, args.pool,
            args.start, args.end, args.cash, chart_dir
        )
        if images:
            log.info(f"📊 可视化图表 ({len(images)}张):")
            for img in images:
                log.info(f"   {img}")
    except Exception as e:
        log.warning(f"可视化生成失败 (不影响回测结果): {e}")

    log.info("🦞 完成!")


if __name__ == "__main__":
    main()
