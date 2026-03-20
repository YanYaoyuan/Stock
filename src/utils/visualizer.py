# 🦞 龙虾量化 — 可视化模块

"""
回测结果可视化 + 选股雷达图
依赖: matplotlib (pip install matplotlib)
"""

import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无头模式 (服务器/TUI环境)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

log = logging.getLogger(__name__)

# 配色
COLORS = {
    'bg': '#1a1a2e',
    'card': '#16213e',
    'green': '#00e676',
    'red': '#ff5252',
    'blue': '#448aff',
    'orange': '#ffab40',
    'purple': '#b388ff',
    'text': '#e0e0e0',
    'grid': '#2a2a4a',
    'profit': '#00c853',
    'loss': '#ff1744',
    'accent': '#ffd740',
}


def set_dark_style():
    """设置深色主题"""
    plt.style.use('dark_background')
    fig = plt.figure()
    fig.patch.set_facecolor(COLORS['bg'])
    plt.close(fig)


def generate_backtest_charts(trades, daily_values, stats, pool_name,
                             start_date, end_date, initial_cash,
                             output_dir, benchmark_values=None):
    """
    生成回测可视化图表

    Args:
        trades: 交易记录列表
        daily_values: 每日净值列表 [{'date':, 'total':, 'cash':, 'positions_value':}]
        stats: 回测统计 dict
        pool_name: 股票池名称
        start_date, end_date: 回测区间
        initial_cash: 初始资金
        output_dir: 输出目录
        benchmark_values: 可选, 基准净值列表 [float, ...] (同长度daily_values)

    Returns:
        生成的图片路径列表
    """
    set_dark_style()
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    images = []

    if not daily_values or len(daily_values) < 2:
        log.warning("数据不足，跳过可视化")
        return images

    dates = [dv['date'] for dv in daily_values]
    total_values = [dv['total'] for dv in daily_values]
    cash_values = [dv['cash'] for dv in daily_values]
    pos_values = [dv['positions_value'] for dv in daily_values]

    # 图1: 权益曲线 + 回撤
    img1 = os.path.join(output_dir, f"equity_curve_{timestamp}.png")
    _plot_equity_curve(dates, total_values, cash_values, pos_values,
                       stats, pool_name, initial_cash, benchmark_values, img1)
    images.append(img1)

    # 图2: 交易分布 + 月度收益
    img2 = os.path.join(output_dir, f"trade_analysis_{timestamp}.png")
    _plot_trade_analysis(trades, dates, total_values, initial_cash, img2)
    images.append(img2)

    # 图3: 核心指标仪表盘
    img3 = os.path.join(output_dir, f"dashboard_{timestamp}.png")
    _plot_dashboard(stats, pool_name, start_date, end_date, initial_cash, img3)
    images.append(img3)

    plt.close('all')
    return images


def generate_screener_chart(stocks, output_dir):
    """
    选股结果可视化 — 多因子评分图

    Args:
        stocks: [{'code', 'name', 'roe', 'growth', 'sentiment_score'}, ...]
        output_dir: 输出目录

    Returns:
        图片路径
    """
    set_dark_style()
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img = os.path.join(output_dir, f"screener_{timestamp}.png")

    if not stocks:
        log.warning("选股结果为空，跳过可视化")
        return None

    _plot_screener(stocks, img)
    plt.close('all')
    return img


# ============================================================
#  内部绑图函数
# ============================================================

def _plot_equity_curve(dates, total_values, cash_values, pos_values,
                       stats, pool_name, initial_cash, benchmark_values, save_path):
    """权益曲线 + 回撤图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                    gridspec_kw={'hspace': 0.15})
    fig.patch.set_facecolor(COLORS['bg'])

    # === 上半: 权益曲线 ===
    # 策略净值
    equity_pct = [(v / initial_cash - 1) * 100 for v in total_values]
    ax1.plot(dates, equity_pct, color=COLORS['green'], linewidth=1.5,
             label=f'策略 ({stats["total_return"]:+.1f}%)', alpha=0.9)

    # 基准净值
    if benchmark_values and len(benchmark_values) == len(dates):
        bench_pct = [(v / benchmark_values[0] - 1) * 100 for v in benchmark_values]
        ax1.plot(dates, bench_pct, color=COLORS['orange'], linewidth=1.2,
                 label='基准 (沪深300)', alpha=0.7, linestyle='--')

    # 现金 vs 持仓
    cash_pct = [(c / initial_cash) * 100 for c in cash_values]
    pos_pct = [(p / initial_cash) * 100 for p in pos_values]
    ax1.fill_between(dates, 0, cash_pct, alpha=0.15, color=COLORS['blue'], label='现金')
    ax1.fill_between(dates, cash_pct, [c + p for c, p in zip(cash_pct, pos_pct)],
                     alpha=0.15, color=COLORS['purple'], label='持仓')

    ax1.axhline(y=0, color=COLORS['grid'], linewidth=0.5, alpha=0.5)
    ax1.set_title(f'🦞 {pool_name} 权益曲线', fontsize=14, color=COLORS['text'], pad=10)
    ax1.set_ylabel('收益率 (%)', color=COLORS['text'])
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.3)
    ax1.grid(True, alpha=0.15, color=COLORS['grid'])
    ax1.tick_params(colors=COLORS['text'])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.set_facecolor(COLORS['card'])

    # === 下半: 回撤图 ===
    peak = initial_cash
    drawdowns = []
    for v in total_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        drawdowns.append(dd)

    ax2.fill_between(dates, drawdowns, 0, alpha=0.6, color=COLORS['red'])
    ax2.plot(dates, drawdowns, color=COLORS['red'], linewidth=0.8, alpha=0.8)
    ax2.set_ylabel('回撤 (%)', color=COLORS['text'])
    ax2.set_xlabel('')
    ax2.grid(True, alpha=0.15, color=COLORS['grid'])
    ax2.tick_params(colors=COLORS['text'])
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax2.set_facecolor(COLORS['card'])
    ax2.invert_yaxis()

    # 标注最大回撤
    max_dd_idx = np.argmax(drawdowns)
    if drawdowns[max_dd_idx] > 1:
        ax2.annotate(f'最大回撤: {drawdowns[max_dd_idx]:.1f}%',
                     xy=(dates[max_dd_idx], drawdowns[max_dd_idx]),
                     xytext=(dates[max_dd_idx], drawdowns[max_dd_idx] - 3),
                     fontsize=9, color=COLORS['red'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=0.8))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    log.info(f"📊 权益曲线图: {save_path}")


def _plot_trade_analysis(trades, dates, total_values, initial_cash, save_path):
    """交易分析图: 盈亏分布 + 月度收益热力图 + 持仓时长分布"""
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(COLORS['bg'])

    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # === 左上: 盈亏分布直方图 ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS['card'])

    if trades:
        pnls_pct = [t['pnl_pct'] for t in trades]
        colors_hist = [COLORS['green'] if p >= 0 else COLORS['red'] for p in pnls_pct]
        n, bins, patches = ax1.hist(pnls_pct, bins=40, color=COLORS['blue'], alpha=0.7, edgecolor='none')
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge >= 0:
                patch.set_facecolor(COLORS['green'])
            else:
                patch.set_facecolor(COLORS['red'])
            patch.set_alpha(0.7)

    ax1.axvline(x=0, color=COLORS['text'], linewidth=0.5, alpha=0.5)
    ax1.set_title('交易盈亏分布 (%)', color=COLORS['text'], fontsize=11)
    ax1.set_xlabel('收益率 (%)', color=COLORS['text'])
    ax1.set_ylabel('次数', color=COLORS['text'])
    ax1.grid(True, alpha=0.15, color=COLORS['grid'])
    ax1.tick_params(colors=COLORS['text'])

    # === 右上: 持仓时长 vs 收益率 散点图 ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(COLORS['card'])

    if trades:
        hold_days = [t['hold_days'] for t in trades]
        pnl_pcts = [t['pnl_pct'] for t in trades]
        colors_scatter = [COLORS['green'] if p >= 0 else COLORS['red'] for p in pnl_pcts]
        ax2.scatter(hold_days, pnl_pcts, c=colors_scatter, alpha=0.6, s=20, edgecolors='none')

    ax2.axhline(y=0, color=COLORS['text'], linewidth=0.5, alpha=0.5)
    ax2.set_title('持仓时长 vs 收益', color=COLORS['text'], fontsize=11)
    ax2.set_xlabel('持仓天数', color=COLORS['text'])
    ax2.set_ylabel('收益率 (%)', color=COLORS['text'])
    ax2.grid(True, alpha=0.15, color=COLORS['grid'])
    ax2.tick_params(colors=COLORS['text'])

    # === 下: 月度收益柱状图 ===
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor(COLORS['card'])

    if trades and dates:
        # 构建月度收益
        monthly = {}
        for t in trades:
            sell_date = t['sell_date']
            if isinstance(sell_date, str):
                month_key = sell_date[:7]
            else:
                month_key = sell_date.strftime('%Y-%m')
            monthly[month_key] = monthly.get(month_key, 0) + t['pnl']

        sorted_months = sorted(monthly.keys())
        month_returns = [(monthly[m] / initial_cash) * 100 for m in sorted_months]

        x = range(len(sorted_months))
        bar_colors = [COLORS['green'] if r >= 0 else COLORS['red'] for r in month_returns]
        ax3.bar(x, month_returns, color=bar_colors, alpha=0.7, width=0.8, edgecolor='none')

        # X轴标签 (隔几个显示一个)
        step = max(1, len(sorted_months) // 12)
        tick_positions = list(range(0, len(sorted_months), step))
        tick_labels = [sorted_months[i] for i in tick_positions]
        ax3.set_xticks(tick_positions)
        ax3.set_xticklabels(tick_labels, rotation=45, fontsize=8)

    ax3.axhline(y=0, color=COLORS['text'], linewidth=0.5, alpha=0.5)
    ax3.set_title('月度盈亏 (¥)', color=COLORS['text'], fontsize=11)
    ax3.set_xlabel('')
    ax3.set_ylabel('盈亏 (¥)', color=COLORS['text'])
    ax3.grid(True, alpha=0.15, color=COLORS['grid'], axis='y')
    ax3.tick_params(colors=COLORS['text'])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    log.info(f"📊 交易分析图: {save_path}")


def _plot_dashboard(stats, pool_name, start_date, end_date, initial_cash, save_path):
    """核心指标仪表盘"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    ax.axis('off')

    # 标题
    ax.text(0.5, 0.95, f'🦞 龙虾量化回测仪表盘', fontsize=18, color=COLORS['accent'],
            ha='center', va='top', fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.89, f'{pool_name} | {start_date} ~ {end_date}',
            fontsize=11, color=COLORS['text'], ha='center', va='top',
            transform=ax.transAxes)

    # 指标卡片
    metrics = [
        ('总收益率', f"{stats['total_return']:+.2f}%",
         COLORS['green'] if stats['total_return'] >= 0 else COLORS['red']),
        ('年化收益', f"{stats['annual_return']:+.2f}%",
         COLORS['green'] if stats['annual_return'] >= 0 else COLORS['red']),
        ('Sharpe', f"{stats['sharpe']:.2f}",
         COLORS['green'] if stats['sharpe'] >= 1 else (COLORS['orange'] if stats['sharpe'] >= 0.5 else COLORS['red'])),
        ('最大回撤', f"{stats['max_drawdown']:.2f}%",
         COLORS['green'] if stats['max_drawdown'] < 15 else (COLORS['orange'] if stats['max_drawdown'] < 25 else COLORS['red'])),
        ('胜率', f"{stats['win_rate']:.1f}%",
         COLORS['green'] if stats['win_rate'] >= 55 else (COLORS['orange'] if stats['win_rate'] >= 45 else COLORS['red'])),
        ('盈亏比', f"{stats['profit_factor']:.2f}",
         COLORS['green'] if stats['profit_factor'] >= 1.5 else (COLORS['orange'] if stats['profit_factor'] >= 1.0 else COLORS['red'])),
        ('总交易', f"{stats['total_trades']}笔",
         COLORS['blue']),
        ('最终权益', f"¥{stats['final_value']:,.0f}",
         COLORS['green'] if stats['final_value'] >= initial_cash else COLORS['red']),
        ('平均持仓', f"{stats['avg_hold']:.1f}天",
         COLORS['blue']),
    ]

    # 布局: 3列 x 3行
    cols = 3
    rows = 3
    card_w = 0.28
    card_h = 0.18
    gap_x = 0.03
    gap_y = 0.03
    start_x = 0.06
    start_y = 0.72

    for i, (label, value, color) in enumerate(metrics):
        col = i % cols
        row = i // cols
        x = start_x + col * (card_w + gap_x)
        y = start_y - row * (card_h + gap_y)

        # 卡片背景
        rect = FancyBboxPatch((x, y), card_w, card_h,
                               boxstyle="round,pad=0.01",
                               facecolor=COLORS['card'],
                               edgecolor=color, linewidth=1.5,
                               alpha=0.8,
                               transform=ax.transAxes)
        ax.add_patch(rect)

        # 标签
        ax.text(x + card_w / 2, y + card_h * 0.65, label,
                fontsize=10, color=COLORS['text'], ha='center', va='center',
                transform=ax.transAxes, alpha=0.7)
        # 数值
        ax.text(x + card_w / 2, y + card_h * 0.3, value,
                fontsize=16, color=color, ha='center', va='center',
                transform=ax.transAxes, fontweight='bold')

    # 底部: 评级
    score = 0
    if stats['annual_return'] > 5: score += 2
    elif stats['annual_return'] > 0: score += 1
    if stats['sharpe'] > 1: score += 2
    elif stats['sharpe'] > 0.5: score += 1
    if stats['max_drawdown'] < 15: score += 2
    elif stats['max_drawdown'] < 25: score += 1
    if stats['win_rate'] > 55: score += 1
    if stats['profit_factor'] > 1.5: score += 2
    elif stats['profit_factor'] > 1.0: score += 1

    if score >= 8:
        grade, grade_color = '⭐ 优秀', COLORS['green']
    elif score >= 5:
        grade, grade_color = '👍 良好', COLORS['blue']
    elif score >= 3:
        grade, grade_color = '😐 一般', COLORS['orange']
    else:
        grade, grade_color = '❌ 较差', COLORS['red']

    ax.text(0.5, 0.08, f'综合评级: {grade} ({score}/9)',
            fontsize=14, color=grade_color, ha='center', va='center',
            transform=ax.transAxes, fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    log.info(f"📊 仪表盘: {save_path}")


def _plot_screener(stocks, save_path):
    """选股结果 — 多因子条形图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(stocks) * 0.15 + 2)))
    fig.patch.set_facecolor(COLORS['bg'])

    # 排序: 按ROE
    sorted_stocks = sorted(stocks, key=lambda x: x.get('roe') or 0, reverse=True)[:30]

    names = [s['name'] for s in sorted_stocks]
    roes = [s.get('roe') or 0 for s in sorted_stocks]
    growths = [s.get('growth') or 0 for s in sorted_stocks]
    sentiments = [s.get('sentiment_score') or 0 for s in sorted_stocks]

    y_pos = range(len(names))

    # 左: ROE + 增长
    ax1 = axes[0]
    ax1.set_facecolor(COLORS['card'])
    ax1.barh(y_pos, roes, height=0.4, color=COLORS['blue'], alpha=0.8, label='ROE(%)', align='edge')
    ax1.barh([y - 0.4 for y in y_pos], growths, height=0.4, color=COLORS['orange'], alpha=0.8, label='增长(%)', align='edge')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('%', color=COLORS['text'])
    ax1.set_title('基本面评分 Top 30', color=COLORS['text'], fontsize=12)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.tick_params(colors=COLORS['text'])
    ax1.grid(True, alpha=0.15, color=COLORS['grid'], axis='x')
    ax1.invert_yaxis()

    # 右: 情绪评分
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['card'])
    sent_colors = [COLORS['green'] if s >= 0.2 else (COLORS['red'] if s <= -0.2 else COLORS['blue'])
                   for s in sentiments]
    ax2.barh(y_pos, sentiments, height=0.6, color=sent_colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('情绪分数', color=COLORS['text'])
    ax2.set_title('新闻情绪评分', color=COLORS['text'], fontsize=12)
    ax2.axvline(x=0, color=COLORS['text'], linewidth=0.5, alpha=0.5)
    ax2.tick_params(colors=COLORS['text'])
    ax2.grid(True, alpha=0.15, color=COLORS['grid'], axis='x')
    ax2.invert_yaxis()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    log.info(f"📊 选股分析图: {save_path}")
