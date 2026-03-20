#!/usr/bin/env python3
"""
🦞 龙虾量化 — 选股入口
用法: python run_screener.py [--pool hs300|zz500]
"""

import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ROE_MIN, PROFIT_GROWTH_MIN, PE_MAX, PB_MAX, OUTPUT_DIR
from src.data.datasource import BaostockDataSource
from src.screener.stock_screener import StockScreener


class ScreenConfig:
    ROE_MIN = ROE_MIN
    PROFIT_GROWTH_MIN = PROFIT_GROWTH_MIN


def main():
    parser = argparse.ArgumentParser(description='🦞 龙虾选股')
    parser.add_argument('--pool', type=str, default='hs300', choices=['hs300', 'zz500'])
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    
    print(f"🦞 龙虾选股 — {args.pool}")
    print(f"   条件: ROE>{ROE_MIN*100:.0f}% 或 增长>{PROFIT_GROWTH_MIN*100:.0f}%")
    print(f"   估值: PE<{PE_MAX}, PB<{PB_MAX}")
    print("=" * 60)
    
    with BaostockDataSource() as ds:
        pool = ds.get_stock_pool(args.pool)
        print(f"   股票池: {len(pool)} 只")
        
        screener = StockScreener(ds, ScreenConfig())
        passed = screener.screen(pool)
        
        print(f"\n{'='*60}")
        print(f"🏆 选股通过: {len(passed)} 只")
        print(f"{'='*60}")
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
        csv_path = os.path.join(out_dir, f"screener_{datetime.now().strftime('%Y%m%d')}.csv")
        pd.DataFrame(passed).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n📄 结果: {csv_path}")


if __name__ == "__main__":
    main()
