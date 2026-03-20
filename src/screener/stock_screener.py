# 🦞 龙虾量化 — 基本面选股引擎

import logging
from datetime import datetime

log = logging.getLogger(__name__)


class StockScreener:
    """基本面选股: ROE + 利润增长 + 估值过滤"""
    
    def __init__(self, datasource, config):
        self.ds = datasource
        self.cfg = config
    
    def screen(self, pool, ref_date=None):
        """
        对股票池进行基本面筛选
        
        Args:
            pool: [{'code': 'sh.600519', 'name': '贵州茅台'}, ...]
            ref_date: 参考日期 (用于确定财报期)
        
        Returns:
            通过筛选的股票列表: [{'code': ..., 'name': ..., 'roe': ..., 'growth': ...}, ...]
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
                passed.append({
                    'code': code,
                    'name': stock['name'],
                    'roe': roe,
                    'growth': growth,
                })
        
        log.info(f"   基本面通过: {len(passed)}/{len(pool)} 只")
        return passed
