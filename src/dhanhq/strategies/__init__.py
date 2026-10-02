"""
Trading strategy helpers built on top of the DhanHQ historical data APIs.

These are research / backtesting utilities. They never place orders.
"""
from .judas_swing import (
    JudasSwingConfig,
    JudasSwingIndicator,
    JudasSwingBacktester,
    JudasSignal,
    JudasTrade,
    BacktestResult,
    NIFTY_50,
    BANK_NIFTY,
    fetch_intraday_candles,
    normalize_candles,
    resample_candles,
)

__all__ = [
    "JudasSwingConfig",
    "JudasSwingIndicator",
    "JudasSwingBacktester",
    "JudasSignal",
    "JudasTrade",
    "BacktestResult",
    "NIFTY_50",
    "BANK_NIFTY",
    "fetch_intraday_candles",
    "normalize_candles",
    "resample_candles",
]
