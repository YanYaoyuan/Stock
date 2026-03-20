# 🦞 龙虾量化 — 回测引擎

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

from ..utils.indicators import calc_all_indicators
from ..data.datasource import BaostockDataSource

log = logging.getLogger(__name__)


class BacktestEngine:
    """量化回测引擎: 基本面选股 + 技术面择时 + 量价情绪 + 风控 + 可选新闻情绪"""

    def __init__(self, config, sentiment_cache=None):
        """
        Args:
            config: 配置对象 (需包含技术面+风控+情绪参数)
            sentiment_cache: 可选, SentimentDataCache实例 (回测用历史情绪数据)
        """
        self.cfg = config
        self.sentiment_cache = sentiment_cache
        self.sentiment_enabled = sentiment_cache is not None and getattr(config, 'SENTIMENT_ENABLED', True)

        if self.sentiment_enabled:
            log.info(f"   📰 情绪因子已启用 ({sentiment_cache.stats()})")
        else:
            if sentiment_cache is not None and not getattr(config, 'SENTIMENT_ENABLED', True):
                log.info("   情绪因子已禁用 (SENTIMENT_ENABLED=False)")
            else:
                log.info("   情绪因子未启用 (未提供sentiment_cache)")

    def run(self, stock_data, start_date, end_date, cash):
        """
        运行回测

        Args:
            stock_data: [{'code': ..., 'name': ..., 'roe': ..., 'growth': ..., 'df': DataFrame}, ...]
            start_date: 回测起始日 (str YYYY-MM-DD)
            end_date: 回测结束日 (str YYYY-MM-DD)
            cash: 初始资金

        Returns:
            (trades, daily_values, stats)
        """
        trades = []
        daily_values = []
        positions = {}

        # 构建索引
        data = {s['code']: s for s in stock_data}

        # 获取交易日
        all_dates = set()
        for s in stock_data:
            all_dates.update(s['df'].index.tolist())
        all_dates = sorted(all_dates)

        if not all_dates:
            log.error("无交易日数据")
            return [], [], self._empty_stats(cash)

        log.info(f"   交易日: {all_dates[0].strftime('%Y-%m-%d')} ~ "
                 f"{all_dates[-1].strftime('%Y-%m-%d')} ({len(all_dates)}天)")

        cfg = self.cfg

        for date_idx, date in enumerate(all_dates):
            if date_idx % 200 == 0 and date_idx > 0:
                log.info(f"   回测进度: {date.strftime('%Y-%m-%d')} ({date_idx}/{len(all_dates)})")

            # 当日市场情绪 (可选)
            market_sentiment = None
            if self.sentiment_enabled:
                market_sentiment = self.sentiment_cache.get_market_sentiment(date)

            # === 卖出检查 ===
            for code in list(positions.keys()):
                pos = positions[code]
                if code not in data:
                    continue
                df = data[code]['df']
                if date not in df.index:
                    continue
                row = df.loc[date]
                if isinstance(row, pd.DataFrame):
                    continue

                price = row['close']
                if price > pos['highest']:
                    pos['highest'] = price
                pos['hold_days'] += 1

                # 情绪卖出检查 (可选)
                sentiment_sell, sentiment_reason = self._check_sentiment_sell(code, date, pos)
                if sentiment_sell:
                    sell_price = price * (1 - cfg.SLIPPAGE)
                    gross = sell_price * pos['shares']
                    fees = gross * cfg.COMMISSION + gross * cfg.STAMP_TAX
                    net = gross - fees
                    pnl = net - pos['cost']
                    pnl_pct = (sell_price - pos['entry_price']) / pos['entry_price'] * 100
                    cash += net

                    trades.append({
                        'code': code, 'name': data[code]['name'],
                        'roe': pos.get('roe'), 'growth': pos.get('growth'),
                        'buy_date': pos['buy_date'], 'sell_date': date,
                        'entry_price': pos['entry_price'], 'exit_price': sell_price,
                        'shares': pos['shares'], 'pnl': pnl, 'pnl_pct': pnl_pct,
                        'reason': sentiment_reason, 'hold_days': pos['hold_days'],
                    })
                    del positions[code]
                    continue

                sell, reason = self._check_sell(price, pos)
                if sell:
                    sell_price = price * (1 - cfg.SLIPPAGE)
                    gross = sell_price * pos['shares']
                    fees = gross * cfg.COMMISSION + gross * cfg.STAMP_TAX
                    net = gross - fees
                    pnl = net - pos['cost']
                    pnl_pct = (sell_price - pos['entry_price']) / pos['entry_price'] * 100
                    cash += net

                    trades.append({
                        'code': code, 'name': data[code]['name'],
                        'roe': pos.get('roe'), 'growth': pos.get('growth'),
                        'buy_date': pos['buy_date'], 'sell_date': date,
                        'entry_price': pos['entry_price'], 'exit_price': sell_price,
                        'shares': pos['shares'], 'pnl': pnl, 'pnl_pct': pnl_pct,
                        'reason': reason, 'hold_days': pos['hold_days'],
                    })
                    del positions[code]

            # === 买入检查 ===
            if len(positions) < cfg.MAX_POSITIONS:
                # 市场情绪过滤
                can_buy = True
                if self.sentiment_enabled and market_sentiment is not None:
                    market_min = getattr(cfg, 'MARKET_SENTIMENT_MIN', -0.2)
                    if market_sentiment < market_min:
                        can_buy = False

                if can_buy:
                    # 按因子分排序优先买入高分股票 (如果有因子分)
                    buy_candidates = []
                    for code, info in data.items():
                        if code in positions:
                            continue
                        df_info = info['df']
                        if date not in df_info.index:
                            continue
                        row = df_info.loc[date]
                        if isinstance(row, pd.DataFrame):
                            continue
                        ok, reason = self._check_buy(row, code, date)
                        if ok:
                            score = info.get('factor_score') or 0
                            buy_candidates.append((score, code, info, row))

                    # 因子分高的优先买
                    buy_candidates.sort(key=lambda x: x[0], reverse=True)
                    for score, code, info, row in buy_candidates:
                        if len(positions) >= cfg.MAX_POSITIONS:
                            break
                        buy_price = row['close'] * (1 + cfg.SLIPPAGE)
                        avail = cash * cfg.POSITION_PCT
                        shares = int(avail / buy_price / 100) * 100
                        if shares < 100:
                            continue
                        cost = buy_price * shares * (1 + cfg.COMMISSION)
                        if cost > cash:
                            continue
                        cash -= cost
                        positions[code] = {
                            'entry_price': buy_price, 'highest': buy_price,
                            'hold_days': 0, 'shares': shares,
                            'buy_date': date, 'cost': cost,
                            'roe': info.get('roe'), 'growth': info.get('growth'),
                            'buy_sentiment': self.sentiment_cache.get_sentiment(code, date) if self.sentiment_enabled else None,
                        }

            # === 每日净值 ===
            pos_val = 0
            for code, pos in positions.items():
                if code in data:
                    df = data[code]['df']
                    if date in df.index:
                        r = df.loc[date]
                        if isinstance(r, pd.Series):
                            pos_val += r['close'] * pos['shares']
            daily_values.append({
                'date': date, 'cash': cash,
                'positions_value': pos_val,
                'total': cash + pos_val,
            })

        stats = self._analyze(trades, daily_values, cash)
        return trades, daily_values, stats

    def _check_buy(self, row, code=None, date=None):
        """买入信号: 估值 + 技术面 + 量价情绪 + 可选新闻情绪"""
        cfg = self.cfg

        # === 估值过滤 ===
        pe, pb = row['peTTM'], row['pbMRQ']
        if pd.isna(pe) or pe <= 0 or pe > cfg.PE_MAX:
            return False, f"PE={pe}"
        if pd.isna(pb) or pb <= 0 or pb > cfg.PB_MAX:
            return False, f"PB={pb}"

        for v in ['MA5', 'MA20', 'MA60', 'DIF', 'DEA', 'RSI', 'VOL_MA5']:
            if pd.isna(row.get(v)):
                return False, "指标不足"

        # === 情绪过滤 (可选) ===
        stock_sentiment = None
        if self.sentiment_enabled and code and date:
            sentiment_data = self.sentiment_cache.get_sentiment(code, date)
            if sentiment_data is not None:
                stock_sentiment = sentiment_data['sentiment_score']
                stock_sentiment_min = getattr(cfg, 'STOCK_SENTIMENT_MIN', -0.3)
                if stock_sentiment < stock_sentiment_min:
                    return False, f"情绪低({stock_sentiment:.2f})"

        # === 技术面趋势 ===
        ma5, ma20, ma60 = row['MA5'], row['MA20'], row['MA60']
        if not (ma5 > ma20 > ma60):
            return False, "非多头"
        if (ma5 - ma20) / ma20 < cfg.MA_SPREAD_MIN:
            return False, "趋势弱"

        # MACD
        if row['DIF'] <= row['DEA']:
            return False, "MACD差"

        # RSI — 情绪好时放宽上限
        rsi = row['RSI']
        rsi_high = cfg.RSI_HIGH
        if self.sentiment_enabled and stock_sentiment is not None:
            boost_threshold = getattr(cfg, 'STOCK_SENTIMENT_BOOST', 0.3)
            if stock_sentiment > boost_threshold:
                rsi_high = min(rsi_high + 5, 75)  # 放宽到70-75
        if not (cfg.RSI_LOW <= rsi <= rsi_high):
            return False, f"RSI={rsi:.0f}"

        # === 量价情绪 (历史代理指标) ===
        if row['volume'] <= row['VOL_MA5'] * cfg.VOL_RATIO_MIN:
            return False, "缩量"

        if hasattr(row, 'pctChg') and not pd.isna(row.get('pctChg')):
            if abs(row['pctChg']) > 7:
                return False, f"涨幅异常{row['pctChg']:.1f}%"

        vol_ratio = row['volume'] / row['VOL_MA5'] if row['VOL_MA5'] > 0 else 1
        if vol_ratio > 3.0:
            return False, f"天量(vol_ratio={vol_ratio:.1f})"

        return True, "OK"

    def _check_sentiment_sell(self, code, date, pos):
        """
        情绪卖出检查 — 买入后情绪骤降时提前卖出

        Returns:
            (should_sell: bool, reason: str)
        """
        if not self.sentiment_enabled:
            return False, ""

        current = self.sentiment_cache.get_sentiment(code, date)
        if current is None:
            return False, ""

        buy_sent = pos.get('buy_sentiment')
        if buy_sent is None:
            return False, ""

        buy_score = buy_sent.get('sentiment_score', 0)
        current_score = current['sentiment_score']

        # 情绪骤降: 从正面跌到负面，降幅超过0.5
        if buy_score > 0.2 and current_score < -0.3 and (buy_score - current_score) > 0.5:
            return True, f"情绪骤降({buy_score:+.2f}→{current_score:+.2f})"

        return False, ""

    def _check_sell(self, price, pos):
        """卖出信号"""
        cfg = self.cfg
        pnl = (price - pos['entry_price']) / pos['entry_price']

        if pnl <= -cfg.STOP_LOSS:
            return True, f"止损({pnl*100:.1f}%)"
        if pnl >= cfg.TAKE_PROFIT:
            return True, f"止盈(+{pnl*100:.1f}%)"

        dd = (price - pos['highest']) / pos['highest']
        if dd <= -cfg.TRAILING_STOP and pnl > cfg.TRAILING_ACTIVATE:
            return True, f"移动止损(回落{-dd*100:.1f}%)"

        if pos['hold_days'] >= cfg.MAX_HOLD_DAYS:
            return True, f"时间({pos['hold_days']}天,{pnl*100:+.1f}%)"

        return False, ""

    def _analyze(self, trades, daily_values, initial_cash):
        """分析回测结果"""
        if not trades:
            return self._empty_stats(initial_cash)

        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]

        final_value = daily_values[-1]['total']
        total_return = (final_value - initial_cash) / initial_cash * 100
        years = (daily_values[-1]['date'] - daily_values[0]['date']).days / 365.25
        annual_return = ((final_value / initial_cash) ** (1/years) - 1) * 100 if years > 0 else 0

        peak = initial_cash
        max_dd = 0
        for dv in daily_values:
            if dv['total'] > peak:
                peak = dv['total']
            dd = (peak - dv['total']) / peak * 100
            if dd > max_dd:
                max_dd = dd

        daily_rets = [(daily_values[i]['total'] - daily_values[i-1]['total']) / daily_values[i-1]['total']
                      for i in range(1, len(daily_values))]
        std = np.std(daily_rets)
        sharpe = np.mean(daily_rets) / std * np.sqrt(252) if std > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades) * 100,
            'avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([t['pnl'] for t in losses]) if losses else 0,
            'profit_factor': sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in losses)) if losses else 0,
            'final_value': final_value,
            'avg_hold': np.mean([t['hold_days'] for t in trades]),
            'total_profit': sum(t['pnl'] for t in wins),
            'total_loss': sum(t['pnl'] for t in losses),
        }

    def _empty_stats(self, cash):
        return {
            'total_return': 0, 'annual_return': 0, 'max_drawdown': 0,
            'sharpe': 0, 'total_trades': 0, 'wins': 0, 'losses': 0,
            'win_rate': 0, 'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
            'final_value': cash, 'avg_hold': 0, 'total_profit': 0, 'total_loss': 0,
        }


def generate_report(trades, daily_values, stats, pool_name, start, end, cash, output_dir):
    """生成Markdown报告"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)

    rpt_dir = os.path.join(output_dir, 'reports')
    rpt = os.path.join(rpt_dir, f"backtest_{datetime.now().strftime('%Y%m%d')}.md")

    years = (daily_values[-1]['date'] - daily_values[0]['date']).days / 365.25 if daily_values else 1

    md = f"# 🦞 龙虾量化回测报告\n\n"
    md += f"> {datetime.now().strftime('%Y-%m-%d %H:%M')} | {pool_name} | {start}~{end}\n\n"

    md += "## 核心指标\n\n"
    md += "| 指标 | 值 |\n|------|-----|\n"
    md += f"| 最终权益 | ¥{stats['final_value']:,.0f} |\n"
    md += f"| 总收益率 | **{stats['total_return']:+.2f}%** |\n"
    md += f"| 年化收益 | **{stats['annual_return']:+.2f}%** |\n"
    md += f"| Sharpe | {stats['sharpe']:.2f} |\n"
    md += f"| 最大回撤 | {stats['max_drawdown']:.2f}% |\n"
    md += f"| 交易 | {stats['total_trades']}笔 (盈{stats['wins']} 亏{stats['losses']}) |\n"
    md += f"| 胜率 | **{stats['win_rate']:.1f}%** |\n"
    md += f"| 盈亏比 | {stats['profit_factor']:.2f} |\n"
    md += f"| 平均持仓 | {stats['avg_hold']:.1f}天 |\n\n"

    if trades:
        md += "---\n\n## 交易明细\n\n"
        md += "| # | 股票 | ROE | 增长 | 买日 | 卖日 | 买入 | 卖出 | 天 | 盈亏 | % | 原因 |\n"
        md += "|---|------|-----|------|------|------|------|------|-----|------|-----|------|\n"
        for i, t in enumerate(trades):
            md += f"| {i+1} | {t['name']} | {t.get('roe','—') or '—'} | {t.get('growth','—') or '—'} | "
            md += f"{t['buy_date'].strftime('%m-%d')} | {t['sell_date'].strftime('%m-%d')} | "
            md += f"{t['entry_price']:.2f} | {t['exit_price']:.2f} | {t['hold_days']} | "
            md += f"{t['pnl']:+.0f} | {t['pnl_pct']:+.1f}% | {t['reason']} |\n"

    md += "\n---\n> ⚠️ 仅供学习研究\n"

    with open(rpt, 'w', encoding='utf-8') as f:
        f.write(md)

    if trades:
        tdf = pd.DataFrame(trades)
        tdf['buy_date'] = tdf['buy_date'].dt.strftime('%Y-%m-%d')
        tdf['sell_date'] = tdf['sell_date'].dt.strftime('%Y-%m-%d')
        tdf.to_csv(os.path.join(rpt_dir, f"trades_{datetime.now().strftime('%Y%m%d')}.csv"),
                   index=False, encoding='utf-8-sig')

    return rpt
