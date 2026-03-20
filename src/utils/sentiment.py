# 🦞 龙虾量化 — 实时新闻与情绪分析模块

"""
功能:
  1. 市场整体情绪: 东方财富7x24财经直播
  2. 个股舆情: Web搜索个股新闻
  3. 资金面异动: 腾讯实时行情接口
  4. 情感分析: 基于关键词的情感打分 (无需API Key)
"""

import re
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from collections import defaultdict

log = logging.getLogger(__name__)

# 情感词典 (扩展版)
POSITIVE_WORDS = [
    # 涨跌
    '大涨', '暴涨', '拉升', '上涨', '走高', '飘红', '翻红', '收红',
    '涨停', '封板', '连板', '突破', '新高', '历史新高', '强势',
    '反弹', '回升', '回暖', '企稳', '触底反弹', 'V型反转',
    # 政策利好
    '利好', '超预期', '超预期增长', '大幅增长', '净利增长', '营收增长',
    '业绩大增', '利润翻倍', '业绩预增', '高增长', '爆发',
    '订单大增', '中标', '获批复', '获批', '签约', '合作',
    '增持', '回购', '抄底', '加仓', '看好', '推荐',
    '降准', '降息', '放水', '宽松', '刺激',
    # 行业景气
    '景气', '高景气', '繁荣', '供不应求', '涨价', '提价',
    '满产', '产能扩张', '扩张', '渗透率提升', '拐点',
    # 资金面
    '北向资金流入', '外资流入', '主力流入', '资金净流入',
    '放量', '放量上涨', '底部放量', '天量',
    '融资买入', '机构买入', '机构调研', '机构加仓',
]

NEGATIVE_WORDS = [
    # 跌
    '大跌', '暴跌', '跳水', '闪崩', '崩盘', '崩塌', '暴跌',
    '跌停', '破位', '破发', '新低', '历史新低', '续创',
    '暴跌', '重挫', '大跌', '下挫', '回落', '下行', '走低',
    '杀跌', '大跌', '大跌', '大跌',
    # 利空
    '利空', '不及预期', '大幅下滑', '净利下滑', '营收下滑',
    '业绩下滑', '业绩变脸', '预亏', '亏损', '巨亏',
    '减持', '清仓', '抛售', '套现', '撤离', '撤退',
    '降级', '看空', '回避', '卖出', '下调',
    '加息', '紧缩', '收紧', '监管', '处罚', '立案',
    # 风险
    '风险', '爆雷', '暴雷', '违约', '退市', 'ST', '暂停上市',
    '商誉减值', '资产减值', '计提', '坏账',
    '停产', '停产整顿', '停产整改', '环保',
    # 资金面
    '北向资金流出', '外资流出', '主力流出', '资金净流出',
    '缩量', '地量', '无量', '阴跌',
    '融资卖出', '机构卖出', '机构减仓', '清仓式减持',
]


class SentimentAnalyzer:
    """基于关键词的情感分析器 (无需LLM API)"""
    
    def __init__(self):
        self.positive = POSITIVE_WORDS
        self.negative = NEGATIVE_WORDS
    
    def analyze_text(self, text):
        """
        分析文本情感
        Returns: {'score': -1到1, 'positive_count': int, 'negative_count': int, 'signals': list}
        """
        if not text:
            return {'score': 0, 'positive_count': 0, 'negative_count': 0, 'signals': []}
        
        pos_found = []
        neg_found = []
        
        for word in self.positive:
            if word in text:
                pos_found.append(word)
        
        for word in self.negative:
            if word in text:
                neg_found.append(word)
        
        total = len(pos_found) + len(neg_found)
        if total == 0:
            score = 0
        else:
            # 加权: 利空词权重略高 (损失厌恶)
            score = (len(pos_found) * 1.0 - len(neg_found) * 1.2) / total
        
        return {
            'score': max(-1, min(1, score)),
            'positive_count': len(pos_found),
            'negative_count': len(neg_found),
            'signals': pos_found[:5] + neg_found[:5],
        }
    
    def analyze_news_list(self, news_items):
        """
        分析一组新闻的情感
        news_items: [{'title': str, 'content': str, 'time': str}, ...]
        Returns: {'overall_score': float, 'news_count': int, 'details': list}
        """
        if not news_items:
            return {'overall_score': 0, 'news_count': 0, 'details': []}
        
        total_score = 0
        details = []
        
        for item in news_items:
            text = item.get('title', '') + ' ' + item.get('content', '')
            result = self.analyze_text(text)
            
            detail = {
                'title': item.get('title', '')[:80],
                'time': item.get('time', ''),
                'score': result['score'],
                'positive': result['positive_count'],
                'negative': result['negative_count'],
                'signals': result['signals'],
            }
            details.append(detail)
            total_score += result['score']
        
        return {
            'overall_score': total_score / len(news_items),
            'news_count': len(news_items),
            'details': details,
        }


class NewsFetcher:
    """新闻抓取器"""
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    def fetch_market_news(self, page_size=10):
        """
        获取市场整体新闻 (东方财富7x24)
        Returns: [{'title': str, 'time': str, 'digest': str}, ...]
        """
        url = "https://np-listapi.eastmoney.com/comm/web/getNewsByColumns"
        params = {
            'client': 'web',
            'biz': 'web_news_col',
            'column': '293',  # 7x24直播
            'order': '1',
            'needInteractData': '0',
            'page_size': str(page_size),
            'page_index': '1',
            'req_trace': f'lobster_{int(time.time())}',
        }
        
        try:
            resp = requests.get(url, params=params, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            items = data.get('data', {}).get('list', [])
            
            result = []
            for item in items:
                result.append({
                    'title': item.get('title', ''),
                    'time': item.get('showTime', ''),
                    'digest': item.get('digest', ''),
                })
            return result
        except Exception as e:
            log.warning(f"市场新闻获取失败: {e}")
            return []
    
    def fetch_stock_news(self, stock_code, stock_name=None):
        """
        获取个股相关新闻 (多源: 东方财富搜索 + 百度新闻)
        stock_code: 'sh.600519' 或 '600519'
        Returns: [{'title': str, 'time': str, 'source': str}, ...]
        """
        clean_code = stock_code.replace('sh.', '').replace('sz.', '').replace('bj.', '')
        keyword = stock_name or clean_code
        
        # 方案1: 东方财富搜索
        result = self._fetch_eastmoney_news(keyword)
        if result:
            return result
        
        # 方案2: 百度新闻搜索
        result = self._fetch_baidu_news(keyword)
        if result:
            return result
        
        return []
    
    def _fetch_eastmoney_news(self, keyword):
        """东方财富新闻搜索"""
        url = "https://search-api-web.eastmoney.com/search/jsonp"
        try:
            params = {
                'cb': 'jQuery_callback',
                'param': json.dumps({
                    "uid": "", "keyword": keyword,
                    "type": ["cmsArticleWebOld"],
                    "client": "web", "clientType": "web", "clientVersion": "curr",
                    "param": {"cmsArticleWebOld": {
                        "searchScope": "default", "sort": "default",
                        "pageIndex": 1, "pageSize": 10, "preTag": ""
                    }}
                })
            }
            resp = requests.get(url, params=params, headers=self.HEADERS, timeout=10)
            if resp.status_code == 200:
                text = resp.text
                if text.startswith('jQuery_callback('):
                    text = text[16:-1]
                data = json.loads(text)
                items = data.get('result', {}).get('cmsArticleWebOld', [])
                if items:
                    return [{
                        'title': item.get('title', ''),
                        'time': item.get('date', ''),
                        'source': '东方财富',
                        'url': item.get('url', ''),
                    } for item in items[:10]]
        except Exception as e:
            log.debug(f"东方财富搜索失败: {e}")
        return []
    
    def _fetch_baidu_news(self, keyword):
        """百度新闻搜索"""
        try:
            url = f"https://news.baidu.com/ns"
            params = {
                'word': f'{keyword} 股票',
                'tn': 'news',
                'from': 'news',
                'cl': '2',
                'rn': '10',
                'ct': '1',
            }
            resp = requests.get(url, params=params, headers=self.HEADERS, timeout=10)
            if resp.status_code == 200:
                # 从HTML中提取新闻标题
                import re
                titles = re.findall(r'<h3[^>]*class="c-title"[^>]*>.*?<a[^>]*>(.*?)</a>', resp.text, re.DOTALL)
                if not titles:
                    titles = re.findall(r'<h3[^>]*>.*?<a[^>]*>(.*?)</a>', resp.text, re.DOTALL)
                
                # 清理HTML标签
                clean_titles = []
                for t in titles[:10]:
                    clean = re.sub(r'<[^>]+>', '', t).strip()
                    if clean and len(clean) > 5:
                        clean_titles.append(clean)
                
                if clean_titles:
                    return [{
                        'title': t,
                        'time': '',
                        'source': '百度新闻',
                        'url': '',
                    } for t in clean_titles]
        except Exception as e:
            log.debug(f"百度新闻搜索失败: {e}")
        return []
    
    def fetch_stock_realtime(self, stock_code):
        """
        获取个股实时行情 (腾讯接口)
        stock_code: 'sh600519' 或 'sz000858'
        Returns: dict with price, change_pct, volume etc.
        """
        try:
            url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            params = {
                'param': f"{stock_code},day,,,3,qfq"
            }
            resp = requests.get(url, params=params, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            key = stock_code
            stock_data = data.get('data', {}).get(key, {})
            qt = stock_data.get('qt', {}).get(key, [])
            
            if not qt:
                return None
            
            # qt字段: [0]=market, 1=name, 2=code, 3=最新价, 4=昨收, 5=今开,
            #   6=最高, 7=成交量(手), 8=成交额, 9=涨跌额, 10=涨跌幅, ...
            return {
                'name': qt[1],
                'code': qt[2],
                'price': float(qt[3]),
                'prev_close': float(qt[4]),
                'change': float(qt[9]),
                'change_pct': float(qt[10]),
                'high': float(qt[6]),
                'low': stock_data.get('qfqday', [[]])[0][3] if stock_data.get('qfqday') else 0,
                'volume': float(qt[7]),
                'amount': float(qt[8]),
            }
        except Exception as e:
            log.debug(f"实时行情获取失败 {stock_code}: {e}")
            return None


class MarketSentimentEngine:
    """
    市场情绪引擎
    结合市场整体新闻 + 个股舆情 + 实时行情 → 综合情绪评分
    """
    
    def __init__(self):
        self.fetcher = NewsFetcher()
        self.analyzer = SentimentAnalyzer()
    
    def get_market_sentiment(self):
        """
        获取市场整体情绪
        Returns: {
            'score': -1~1,
            'positive_news': int,
            'negative_news': int,
            'hot_topics': list,
            'timestamp': str,
        }
        """
        news = self.fetcher.fetch_market_news(page_size=20)
        result = self.analyzer.analyze_news_list(news)
        
        positive = sum(1 for d in result['details'] if d['score'] > 0.2)
        negative = sum(1 for d in result['details'] if d['score'] < -0.2)
        
        # 热门话题
        hot_topics = []
        for d in result['details']:
            if abs(d['score']) > 0.3:
                hot_topics.append({
                    'topic': d['title'],
                    'sentiment': 'positive' if d['score'] > 0 else 'negative',
                    'score': d['score'],
                })
        
        return {
            'score': result['overall_score'],
            'news_count': result['news_count'],
            'positive_news': positive,
            'negative_news': negative,
            'hot_topics': hot_topics[:5],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    
    def get_stock_sentiment(self, stock_code, stock_name=None):
        """
        获取个股情绪
        Returns: {
            'code': str,
            'name': str,
            'news_sentiment': float (-1~1),
            'realtime': dict or None,
            'news_count': int,
            'signals': list,
            'timestamp': str,
        }
        """
        # 新闻情感
        news = self.fetcher.fetch_stock_news(stock_code, stock_name)
        news_result = self.analyzer.analyze_news_list(news) if news else {'overall_score': 0, 'news_count': 0, 'details': []}
        
        # 实时行情
        # 格式化代码: sh600519
        clean_code = stock_code.replace('.', '')
        if stock_code.startswith('sh'):
            qt_code = f"sh{clean_code}"
        elif stock_code.startswith('sz'):
            qt_code = f"sz{clean_code}"
        else:
            qt_code = stock_code
        
        realtime = self.fetcher.fetch_stock_realtime(qt_code)
        
        # 综合情绪: 新闻60% + 行情40%
        news_score = news_result['overall_score']
        
        if realtime:
            # 行情情绪: 基于涨跌幅
            if realtime['change_pct'] > 3:
                market_score = 0.8
            elif realtime['change_pct'] > 1:
                market_score = 0.4
            elif realtime['change_pct'] > 0:
                market_score = 0.2
            elif realtime['change_pct'] > -1:
                market_score = -0.2
            elif realtime['change_pct'] > -3:
                market_score = -0.5
            else:
                market_score = -0.8
        else:
            market_score = 0
        
        # 如果没有新闻数据, 行情权重更高
        if news_result['news_count'] == 0:
            combined = market_score
        else:
            combined = news_score * 0.6 + market_score * 0.4
        
        # 收集信号词
        all_signals = []
        for d in news_result['details']:
            if d['signals']:
                all_signals.extend(d['signals'])
        
        return {
            'code': stock_code,
            'name': stock_name or (realtime.get('name', '') if realtime else ''),
            'sentiment_score': combined,
            'news_sentiment': news_score,
            'market_sentiment': market_score,
            'realtime': realtime,
            'news_count': news_result['news_count'],
            'signals': list(set(all_signals))[:10],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    
    def scan_watchlist(self, stock_list):
        """
        批量扫描自选股情绪
        stock_list: [{'code': 'sh.600519', 'name': '贵州茅台'}, ...]
        Returns: list of sentiment results
        """
        results = []
        for i, stock in enumerate(stock_list):
            try:
                result = self.get_stock_sentiment(stock['code'], stock['name'])
                results.append(result)
                # 避免请求太快
                if i < len(stock_list) - 1:
                    time.sleep(0.5)
            except Exception as e:
                log.warning(f"情绪扫描失败 {stock['code']}: {e}")
                results.append({
                    'code': stock['code'], 'name': stock['name'],
                    'sentiment_score': 0, 'error': str(e),
                })
        
        # 按情绪排序
        results.sort(key=lambda x: x.get('sentiment_score', 0), reverse=True)
        return results
