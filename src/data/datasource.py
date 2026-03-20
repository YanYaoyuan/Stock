# 🦞 龙虾量化 — baostock数据接口

import baostock as bs
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class BaostockDataSource:
    """baostock数据源封装"""
    
    def __init__(self):
        self._logged_in = False
    
    def login(self):
        if not self._logged_in:
            rs = bs.login()
            if rs.error_code != '0':
                raise ConnectionError(f"baostock登录失败: {rs.error_msg}")
            self._logged_in = True
            log.info("baostock连接成功")
    
    def logout(self):
        if self._logged_in:
            bs.logout()
            self._logged_in = False
    
    def get_stock_pool(self, pool_type='hs300'):
        """获取股票池"""
        if pool_type == 'hs300':
            rs = bs.query_hs300_stocks()
        elif pool_type == 'zz500':
            rs = bs.query_zz500_stocks()
        else:
            raise ValueError(f"未知股票池: {pool_type}")
        
        pool = []
        while rs.next():
            r = rs.get_row_data()
            pool.append({'code': r[1], 'name': r[2]})
        return pool
    
    def get_kline(self, code, start_date, end_date):
        """获取日K线数据(前复权, 含PE/PB)"""
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume,pctChg,peTTM,pbMRQ,isST",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2"
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        
        if not rows:
            return None
        
        df = pd.DataFrame(rows, columns=rs.fields)
        for c in ['open', 'high', 'low', 'close', 'volume', 'peTTM', 'pbMRQ', 'pctChg']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        # 过滤ST股 (isST=0是正常股)
        df = df[df['isST'] == '0'].copy()
        return df.set_index('date').sort_index()
    
    def get_fundamental(self, code, year, quarter):
        """获取基本面数据 (ROE + 净利润增长)"""
        result = {'roe': None, 'profit_growth': None}
        
        # ROE
        rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if rows:
            roe = pd.to_numeric(rows[0][3], errors='coerce')  # roeAvg
            if not pd.isna(roe):
                result['roe'] = roe * 100  # 转百分比
        
        # 净利润同比增长
        rs2 = bs.query_growth_data(code=code, year=year, quarter=quarter)
        rows2 = []
        while rs2.next():
            rows2.append(rs2.get_row_data())
        if rows2:
            yoy = pd.to_numeric(rows2[0][5], errors='coerce')  # YOYNI
            if not pd.isna(yoy):
                result['profit_growth'] = yoy * 100
        
        return result
    
    def __enter__(self):
        self.login()
        return self
    
    def __exit__(self, *args):
        self.logout()
