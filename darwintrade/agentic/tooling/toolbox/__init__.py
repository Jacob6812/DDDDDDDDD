from __future__ import annotations

from .fundamentals import (
    get_balance_sheet,
    get_cashflow,
    get_filing_text,
    get_income_statement,
)
from .macro import (
    conflict_density_check,
    evidence_coverage_check,
    trace_consistency_check,
)
from .akshare_adapter import (
    stock_us_daily as akshare_stock_us_daily,
    stock_us_famous_spot_em as akshare_stock_us_famous_spot_em,
    stock_us_hist_min_em as akshare_stock_us_hist_min_em,
    stock_value_em as akshare_stock_value_em,
)
from .market import get_indicators, get_stock_data
from .news import get_global_news, get_insider_transactions, get_news

DEFAULT_TOOLBOX = {
    "market.get_stock_data": get_stock_data,
    "market.get_indicators": get_indicators,
    "fundamentals.get_balance_sheet": get_balance_sheet,
    "fundamentals.get_cashflow": get_cashflow,
    "fundamentals.get_income_statement": get_income_statement,
    "fundamentals.get_filing_text": get_filing_text,
    "news.get_news": get_news,
    "news.get_global_news": get_global_news,
    "news.get_insider_transactions": get_insider_transactions,
    "macro.evidence_coverage_check": evidence_coverage_check,
    "macro.conflict_density_check": conflict_density_check,
    "macro.trace_consistency_check": trace_consistency_check,
    "akshare.stock_us_daily": akshare_stock_us_daily,
    "akshare.stock_us_famous_spot_em": akshare_stock_us_famous_spot_em,
    "akshare.stock_us_hist_min_em": akshare_stock_us_hist_min_em,
    "akshare.stock_value_em": akshare_stock_value_em,
}

__all__ = ["DEFAULT_TOOLBOX"]
