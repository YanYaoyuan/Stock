#!/usr/bin/env python3
"""
🦞 龙虾量化 — 回测入口
用法: python run_backtest.py [--pool hs300|zz500] [--start 2020-01-01] [--end 2025-12-31] [--cash 100000]
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
    COMMISSION, STAMP_TAX, SLIPPAGE, OUTPUT_DIR
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


def main():
    parser = argparse.ArgumentParser(description='🦞 龙虾回测系统')
    parser.add_argument('--pool', type=str, default='hs300', choices=['hs300', 'zz500'])
    parser.add_argument('--start', type=str, default='2020-01-01')
    parser.add_argument('--end', type=str, default='2025-12-31')
    parser.add_argument('--cash', type=float, default=100000)
    parser.add_argument('--output', type=str, default=None)
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
    
    log.info("=" * 70)
    log.info("🦞 龙虾 A股量化交易系统")
    log.info("=" * 70)
    log.info(f"   选股: ROE>{ROE_MIN*100:.0f}% 或 增长>{PROFIT_GROWTH_MIN*100:.0f}%")
    log.info(f"   择时: MA多头 + MACD + RSI 30-65 + 放量")
    log.info(f"   风控: 止损{STOP_LOSS*100}% | 止盈{TAKE_PROFIT*100}% | 移动止损{TRAILING_STOP*100}% | 时间{MAX_HOLD_DAYS}天")
    log.info(f"   区间: {args.start} ~ {args.end} | 资金: ¥{args.cash:,.0f}")
    log.info("=" * 70)
    
    cfg = Config()
    
    with BaostockDataSource() as ds:
        # Step 1: 获取股票池
        pool = ds.get_stock_pool(args.pool)
        log.info(f"   股票池: {args.pool} ({len(pool)}只)")
        
        # Step 2: 基本面选股
        screener = StockScreener(ds, cfg)
        screened = screener.screen(pool)
        
        if not screened:
            log.error("❌ 选股通过为0，无法回测")
            return
        
        # Step 3: 下载K线 + 计算指标
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
                        'df': df,
                    })
            except Exception as e:
                log.warning(f"   K线失败 {s['code']}: {e}")
        
        log.info(f"   K线加载: {len(stock_data)} 只")
        
        if not stock_data:
            log.error("❌ 无K线数据")
            return
        
        # Step 4: 回测
        engine = BacktestEngine(cfg)
        trades, daily_values, stats = engine.run(stock_data, args.start, args.end, args.cash)
    
    # 打印结果
    print(f"\n{'='*70}")
    print(f"🦞 回测结果 — {args.pool}")
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
    log.info("🦞 完成!")


if __name__ == "__main__":
    main()
