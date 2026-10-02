"""
ICT Judas Swing ("Morning Trap") indicator and backtester for Indian index / equity
5-minute candles (NSE opens 09:15 IST).

Strategy summary
----------------
In the first 15-30 minutes after the 09:15 open, smart money often pushes price in a
*false* direction to run the stop losses sitting beyond obvious liquidity levels
(Previous Day High / Low and the Opening Range High / Low). Once that liquidity is
swept and a single 5-minute reversal candle closes back inside, the *real*
directional move of the day begins in the opposite direction.

Bullish Judas Swing (the classic Nifty morning trap)
    1. 09:15 open, price expands down quickly (looks bearish).
    2. A candle's LOW trades below PDL / Opening-Range-Low  -> liquidity grab.
    3. A GREEN 5-minute candle closes back ABOVE the swept level -> reversal.
    4. Go LONG at the open of the next candle.
       Stop  = lowest low of the sweep - buffer.
       Target = fixed points (Nifty: 30-50 points quick scalp).

Bearish Judas Swing is the mirror image (sweep PDH / ORH, red reversal candle, short).

Nothing in this module places orders. It only reads candles (via the DhanHQ
historical API or any DataFrame / CSV you supply) and produces signals + a backtest.

Typical use::

    from dhanhq import DhanContext, dhanhq
    from dhanhq.strategies import JudasSwingBacktester, JudasSwingConfig, NIFTY_50

    dhan = dhanhq(DhanContext(client_id, access_token))
    bt = JudasSwingBacktester(dhan, **NIFTY_50, config=JudasSwingConfig(target_points=40))
    result = bt.run("2025-08-01", "2025-09-30")
    print(result.summary())
    print(result.trades)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

IST = "Asia/Kolkata"
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

#: Dhan instrument descriptors for the two most traded NSE indices.
NIFTY_50 = {"security_id": "13", "exchange_segment": "IDX_I", "instrument_type": "INDEX"}
BANK_NIFTY = {"security_id": "25", "exchange_segment": "IDX_I", "instrument_type": "INDEX"}

LONG = "LONG"
SHORT = "SHORT"

TRADE_COLUMNS = [
    "date", "direction", "swept_level", "level_price", "sweep_time", "reversal_time", "entry_time",
    "entry", "stop", "target", "risk_points", "exit_time", "exit", "exit_reason", "pnl_points",
    "day_open", "pdh", "pdl", "orh", "orl",
]


# --------------------------------------------------------------------------------------
# Configuration & result containers
# --------------------------------------------------------------------------------------
@dataclass
class JudasSwingConfig:
    """Tunable parameters of the Judas Swing setup. All times are IST."""

    #: Candle size in minutes the indicator operates on (candles are resampled if needed).
    candle_minutes: int = 5
    #: The liquidity sweep must START inside the first N minutes after 09:15.
    judas_window_minutes: int = 30
    #: Opening range = first N minutes of the session (ORH / ORL liquidity).
    opening_range_minutes: int = 15
    #: After the sweep candle, the reversal candle must appear within this many candles
    #: (1 = the sweep candle itself may be the reversal candle).
    reversal_max_candles: int = 3
    #: No new entries after this time (the Judas move is a morning phenomenon).
    entry_cutoff: time = time(10, 15)
    #: Time-based square-off if neither stop nor target was hit.
    exit_time: time = time(15, 15)
    #: Fixed profit target in index points (Nifty: 30-50).
    target_points: float = 40.0
    #: Stop is placed this many points beyond the sweep extreme.
    stop_buffer_points: float = 5.0
    #: Skip the trade if the initial risk (entry -> stop) is bigger than this. None = no limit.
    max_stop_points: Optional[float] = 60.0
    #: "long", "short" or "both".
    direction: str = "both"
    #: Liquidity levels eligible to be swept: any subset of PDH, PDL, ORH, ORL.
    levels: Sequence[str] = ("PDH", "PDL", "ORH", "ORL")
    #: Minimum distance (points) price must travel away from the day open before the sweep
    #: counts as a "bearish / bullish expansion". 0 = only require the sweep itself.
    min_expansion_points: float = 0.0

    def __post_init__(self):
        if self.direction not in ("long", "short", "both"):
            raise ValueError("direction must be 'long', 'short' or 'both'")
        bad = [lvl for lvl in self.levels if lvl not in ("PDH", "PDL", "ORH", "ORL")]
        if bad:
            raise ValueError(f"Unknown liquidity level(s): {bad}")
        if self.candle_minutes <= 0 or self.judas_window_minutes <= 0:
            raise ValueError("candle_minutes and judas_window_minutes must be positive")

    @property
    def judas_window_end(self) -> time:
        return _add_minutes(MARKET_OPEN, self.judas_window_minutes)

    @property
    def opening_range_end(self) -> time:
        return _add_minutes(MARKET_OPEN, self.opening_range_minutes)


@dataclass
class JudasSignal:
    """One Judas Swing setup detected on a trading day (before execution)."""

    trade_date: date
    direction: str                  # LONG / SHORT
    swept_level_name: str           # PDL / ORL / PDH / ORH
    swept_level: float
    sweep_time: datetime            # candle that took out the level
    sweep_extreme: float            # lowest low (long) / highest high (short) of the sweep
    reversal_time: datetime         # the single 5-min reversal candle
    reversal_close: float
    entry_time: datetime            # open of the candle after the reversal candle
    entry_price: float
    stop_price: float
    target_price: float
    day_open: float
    pdh: Optional[float] = None
    pdl: Optional[float] = None
    orh: Optional[float] = None
    orl: Optional[float] = None

    @property
    def risk_points(self) -> float:
        return abs(self.entry_price - self.stop_price)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JudasTrade:
    """An executed (simulated) trade."""

    signal: JudasSignal
    exit_time: datetime
    exit_price: float
    exit_reason: str                # TARGET / STOP / TIME / EOD
    pnl_points: float

    def to_row(self) -> dict:
        s = self.signal
        return {
            "date": s.trade_date,
            "direction": s.direction,
            "swept_level": s.swept_level_name,
            "level_price": s.swept_level,
            "sweep_time": s.sweep_time,
            "reversal_time": s.reversal_time,
            "entry_time": s.entry_time,
            "entry": s.entry_price,
            "stop": s.stop_price,
            "target": s.target_price,
            "risk_points": round(s.risk_points, 2),
            "exit_time": self.exit_time,
            "exit": self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl_points": round(self.pnl_points, 2),
            "day_open": s.day_open,
            "pdh": s.pdh,
            "pdl": s.pdl,
            "orh": s.orh,
            "orl": s.orl,
        }


@dataclass
class BacktestResult:
    """Output of :meth:`JudasSwingBacktester.run`."""

    config: JudasSwingConfig
    from_date: date
    to_date: date
    trades: pd.DataFrame
    signals: List[JudasSignal] = field(default_factory=list)
    trading_days: int = 0
    #: Per-candle indicator frame (levels + signal flags) for charting.
    indicator: Optional[pd.DataFrame] = None

    def stats(self) -> Dict[str, object]:
        return compute_stats(self.trades, self.trading_days)

    def summary(self) -> str:
        s = self.stats()
        lines = [
            f"Judas Swing backtest {self.from_date} -> {self.to_date}",
            f"  direction={self.config.direction} target={self.config.target_points} "
            f"stop_buffer={self.config.stop_buffer_points} window={self.config.judas_window_minutes}m",
            f"  trading days      : {s['trading_days']}",
            f"  trades            : {s['trades']}  (long {s['long_trades']}, short {s['short_trades']})",
            f"  win rate          : {s['win_rate_pct']:.1f}%  ({s['wins']}W / {s['losses']}L)",
            f"  total points      : {s['total_points']:+.2f}",
            f"  avg points/trade  : {s['avg_points']:+.2f}",
            f"  avg win / avg loss: {s['avg_win']:+.2f} / {s['avg_loss']:+.2f}",
            f"  profit factor     : {s['profit_factor']:.2f}",
            f"  max drawdown (pts): {s['max_drawdown_points']:.2f}",
            f"  exits             : {s['exit_reasons']}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Candle helpers
# --------------------------------------------------------------------------------------
def _add_minutes(t: time, minutes: int) -> time:
    return (datetime.combine(date(2000, 1, 1), t) + timedelta(minutes=minutes)).time()


def _session_score(index: pd.DatetimeIndex) -> float:
    """
    How plausible a timestamp reading is: fraction of candles inside NSE market hours,
    plus a bonus when candles sit exactly on the 09:15 open.
    """
    if len(index) == 0:
        return 0.0
    times = index.time
    inside = sum(1 for t in times if MARKET_OPEN <= t < MARKET_CLOSE)
    at_open = any(t == MARKET_OPEN for t in times)
    return inside / len(index) + (1.0 if at_open else 0.0)


def _epoch_to_ist(values: pd.Series, epoch_is_ist: Optional[bool]) -> pd.DatetimeIndex:
    """
    Dhan returns epoch seconds. Depending on the API/version the epoch may be true UTC
    or IST wall-clock encoded as if it were UTC. Auto-detect by checking which reading
    puts the candles inside 09:15-15:30.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    as_utc = pd.DatetimeIndex(pd.to_datetime(numeric, unit="s", utc=True)).tz_convert(IST)
    as_ist = pd.DatetimeIndex(pd.to_datetime(numeric, unit="s")).tz_localize(IST)
    if epoch_is_ist is True:
        return as_ist
    if epoch_is_ist is False:
        return as_utc
    return as_ist if _session_score(as_ist) > _session_score(as_utc) else as_utc


def normalize_candles(data, epoch_is_ist: Optional[bool] = None) -> pd.DataFrame:
    """
    Convert a Dhan intraday response (``response['data']`` - columnar dict), a list of
    candle dicts, or a DataFrame into a clean OHLC DataFrame indexed by IST timestamps.

    Accepted timestamp sources: a ``timestamp``/``start_Time``/``time``/``datetime``/``date``
    column (epoch seconds or parseable strings) or a DatetimeIndex.
    """
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
        data = data["data"]
    df = pd.DataFrame(data).copy()
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df.columns = [str(c).strip().lower() for c in df.columns]
    ts_col = next((c for c in ("timestamp", "start_time", "time", "datetime", "date") if c in df.columns), None)

    if ts_col is not None:
        raw = df[ts_col]
        if pd.api.types.is_numeric_dtype(raw) or pd.to_numeric(raw, errors="coerce").notna().all():
            idx = _epoch_to_ist(raw, epoch_is_ist)
        else:
            parsed = pd.to_datetime(raw)
            idx = pd.DatetimeIndex(parsed)
            idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
        df = df.drop(columns=[ts_col])
        df.index = idx
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(IST) if df.index.tz is None else df.index.tz_convert(IST)
    else:
        raise ValueError("Candle data needs a timestamp column or a DatetimeIndex")

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Candle data is missing the '{col}' column")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["open", "high", "low", "close"])


def resample_candles(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample finer candles to ``minutes`` buckets anchored at 09:15 (NSE open)."""
    if df.empty:
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(f"{minutes}min", origin="start_day", offset="9h15min", label="left", closed="left").agg(agg)
    return out.dropna(subset=["open"])


def _candle_minutes(df: pd.DataFrame) -> Optional[int]:
    if len(df) < 2:
        return None
    deltas = pd.Series(df.index[1:] - df.index[:-1])
    return int(deltas.mode().iloc[0].total_seconds() // 60)


def fetch_intraday_candles(dhan, security_id: str, exchange_segment: str, instrument_type: str,
                           from_date, to_date, interval: int = 5, chunk_days: int = 75,
                           epoch_is_ist: Optional[bool] = None) -> pd.DataFrame:
    """
    Download intraday candles from Dhan for an arbitrary date range by chunking the
    requests (the API serves a limited number of days per call).

    Returns a normalized IST-indexed OHLC DataFrame.
    """
    start = pd.Timestamp(from_date).date()
    end = pd.Timestamp(to_date).date()
    frames = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        resp = dhan.intraday_minute_data(
            security_id=str(security_id),
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
            from_date=cursor.strftime("%Y-%m-%d"),
            to_date=chunk_end.strftime("%Y-%m-%d"),
            interval=interval,
        )
        if not isinstance(resp, dict) or resp.get("status") != "success":
            raise RuntimeError(f"Dhan intraday_minute_data failed for {cursor}..{chunk_end}: {resp}")
        chunk = normalize_candles(resp.get("data") or {}, epoch_is_ist=epoch_is_ist)
        if not chunk.empty:
            frames.append(chunk)
        else:
            logger.warning("No intraday candles returned for %s..%s", cursor, chunk_end)
        cursor = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


# --------------------------------------------------------------------------------------
# Indicator
# --------------------------------------------------------------------------------------
class JudasSwingIndicator:
    """
    Detects ICT Judas Swing setups on 5-minute candles.

    :meth:`compute` returns a per-candle DataFrame with the liquidity levels
    (pdh / pdl / orh / orl), sweep flags and entry flags - handy for plotting.
    :meth:`signals` returns the list of :class:`JudasSignal` (one per day at most).
    """

    def __init__(self, config: Optional[JudasSwingConfig] = None):
        self.config = config or JudasSwingConfig()

    # ---- public API -------------------------------------------------------------------
    def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Normalize + resample candles to the configured candle size and market hours."""
        df = normalize_candles(candles)
        if df.empty:
            return df
        actual = _candle_minutes(df)
        if actual is not None and actual < self.config.candle_minutes:
            df = resample_candles(df, self.config.candle_minutes)
        times = df.index.time
        mask = [(MARKET_OPEN <= t < MARKET_CLOSE) for t in times]
        return df[mask]

    def signals(self, candles: pd.DataFrame) -> List[JudasSignal]:
        df = self.prepare(candles)
        out: List[JudasSignal] = []
        prev_high = prev_low = None
        for day, day_df in df.groupby(df.index.date):
            sig = self._scan_day(day, day_df, prev_high, prev_low)
            if sig is not None:
                out.append(sig)
            prev_high, prev_low = float(day_df["high"].max()), float(day_df["low"].min())
        return out

    def compute(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Per-candle indicator frame: levels, sweep flags, entries, stops and targets."""
        df = self.prepare(candles).copy()
        for col in ("pdh", "pdl", "orh", "orl", "stop", "target"):
            df[col] = float("nan")
        for col in ("sweep_low", "sweep_high", "long_entry", "short_entry"):
            df[col] = False
        df["signal"] = ""
        prev_high = prev_low = None
        for day, day_df in df.groupby(df.index.date):
            idx = day_df.index
            orh, orl, or_ready_at = self._opening_range(day_df)
            df.loc[idx, "pdh"] = prev_high if prev_high is not None else float("nan")
            df.loc[idx, "pdl"] = prev_low if prev_low is not None else float("nan")
            if or_ready_at is not None:
                after_or = idx[idx >= or_ready_at]
                df.loc[after_or, "orh"] = orh
                df.loc[after_or, "orl"] = orl
            sig = self._scan_day(day, day_df, prev_high, prev_low)
            if sig is not None:
                flag = "sweep_low" if sig.direction == LONG else "sweep_high"
                df.loc[sig.sweep_time, flag] = True
                df.loc[sig.entry_time, "long_entry" if sig.direction == LONG else "short_entry"] = True
                df.loc[sig.entry_time, "signal"] = f"{sig.direction}:{sig.swept_level_name}"
                df.loc[sig.entry_time, "stop"] = sig.stop_price
                df.loc[sig.entry_time, "target"] = sig.target_price
            prev_high, prev_low = float(day_df["high"].max()), float(day_df["low"].min())
        return df

    # ---- internals --------------------------------------------------------------------
    def _opening_range(self, day_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[pd.Timestamp]]:
        """(ORH, ORL, first timestamp at which the range is known)."""
        or_end = self.config.opening_range_end
        in_range = day_df[[t < or_end for t in day_df.index.time]]
        after = day_df[[t >= or_end for t in day_df.index.time]]
        if in_range.empty or after.empty:
            return None, None, None
        return float(in_range["high"].max()), float(in_range["low"].min()), after.index[0]

    def _levels_for(self, direction: str, day_open: float, prev_high, prev_low, orh, orl) -> List[Tuple[str, float]]:
        cfg = self.config
        levels: List[Tuple[str, float]] = []
        if direction == LONG:
            if "PDL" in cfg.levels and prev_low is not None and prev_low <= day_open:
                levels.append(("PDL", prev_low))
            if "ORL" in cfg.levels and orl is not None:
                levels.append(("ORL", orl))
        else:
            if "PDH" in cfg.levels and prev_high is not None and prev_high >= day_open:
                levels.append(("PDH", prev_high))
            if "ORH" in cfg.levels and orh is not None:
                levels.append(("ORH", orh))
        return levels

    def _scan_day(self, day, day_df: pd.DataFrame, prev_high, prev_low) -> Optional[JudasSignal]:
        cfg = self.config
        if day_df.empty or day_df.index[0].time() != MARKET_OPEN:
            return None  # partial day / data gap at the open: skip
        orh, orl, or_ready_at = self._opening_range(day_df)
        day_open = float(day_df["open"].iloc[0])
        candidates: List[JudasSignal] = []
        directions = [LONG, SHORT] if cfg.direction == "both" else [LONG if cfg.direction == "long" else SHORT]
        for direction in directions:
            sig = self._scan_direction(day, day_df, direction, day_open, prev_high, prev_low, orh, orl, or_ready_at)
            if sig is not None:
                candidates.append(sig)
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.entry_time)  # whichever triggers first

    def _scan_direction(self, day, day_df, direction, day_open, prev_high, prev_low, orh, orl, or_ready_at):
        cfg = self.config
        levels = self._levels_for(direction, day_open, prev_high, prev_low, orh, orl)
        if not levels:
            return None
        window_end = cfg.judas_window_end
        lows, highs = day_df["low"].values, day_df["high"].values
        opens, closes = day_df["open"].values, day_df["close"].values
        n = len(day_df)

        for i in range(n):
            ts = day_df.index[i]
            if ts.time() >= window_end:
                break
            swept = self._swept_levels(direction, lows[i], highs[i], day_open, levels, ts, or_ready_at)
            if not swept:
                continue
            name, level = swept
            # Reversal: first candle (i..i+reversal_max_candles-1) that is green (long) / red (short)
            # and closes back on the right side of the swept level.
            for j in range(i, min(i + cfg.reversal_max_candles, n)):
                if direction == LONG:
                    reversed_ = closes[j] > opens[j] and closes[j] > level
                else:
                    reversed_ = closes[j] < opens[j] and closes[j] < level
                if not reversed_:
                    continue
                if j + 1 >= n or day_df.index[j + 1].time() >= cfg.entry_cutoff:
                    return None
                return self._build_signal(day, day_df, direction, name, level, i, j, day_open,
                                          prev_high, prev_low, orh, orl)
        return None

    def _swept_levels(self, direction, low, high, day_open, levels, ts, or_ready_at):
        """Return (name, price) of the deepest level swept by this candle, or None."""
        cfg = self.config
        hits = []
        for name, level in levels:
            if name in ("ORH", "ORL") and (or_ready_at is None or ts < or_ready_at):
                continue  # opening range not formed yet
            if direction == LONG and low < level and (day_open - low) >= cfg.min_expansion_points:
                hits.append((name, level))
            elif direction == SHORT and high > level and (high - day_open) >= cfg.min_expansion_points:
                hits.append((name, level))
        if not hits:
            return None
        return min(hits, key=lambda h: h[1]) if direction == LONG else max(hits, key=lambda h: h[1])

    def _build_signal(self, day, day_df, direction, name, level, i, j, day_open,
                      prev_high, prev_low, orh, orl) -> Optional[JudasSignal]:
        cfg = self.config
        entry_ts = day_df.index[j + 1]
        entry = float(day_df["open"].iloc[j + 1])
        if direction == LONG:
            extreme = float(day_df["low"].iloc[i:j + 1].min())
            stop = extreme - cfg.stop_buffer_points
            target = entry + cfg.target_points
        else:
            extreme = float(day_df["high"].iloc[i:j + 1].max())
            stop = extreme + cfg.stop_buffer_points
            target = entry - cfg.target_points
        if cfg.max_stop_points is not None and abs(entry - stop) > cfg.max_stop_points:
            logger.debug("%s %s skipped: risk %.1f > max_stop_points", day, direction, abs(entry - stop))
            return None
        return JudasSignal(
            trade_date=day, direction=direction, swept_level_name=name, swept_level=float(level),
            sweep_time=day_df.index[i].to_pydatetime(), sweep_extreme=extreme,
            reversal_time=day_df.index[j].to_pydatetime(), reversal_close=float(day_df["close"].iloc[j]),
            entry_time=entry_ts.to_pydatetime(), entry_price=entry, stop_price=round(stop, 2),
            target_price=round(target, 2), day_open=day_open,
            pdh=prev_high, pdl=prev_low, orh=orh, orl=orl,
        )


# --------------------------------------------------------------------------------------
# Backtester
# --------------------------------------------------------------------------------------
def simulate_trade(signal: JudasSignal, day_df: pd.DataFrame, exit_time: time) -> JudasTrade:
    """
    Walk candles from the entry candle onward. Stop is checked before target inside a
    candle (conservative). Time exit at ``exit_time`` close, else last candle close.
    """
    is_long = signal.direction == LONG
    entry_ts = pd.Timestamp(signal.entry_time)
    path = day_df[day_df.index >= entry_ts]
    last_ts, last_close = path.index[-1], float(path["close"].iloc[-1])
    for ts, row in path.iterrows():
        hi, lo, close = float(row["high"]), float(row["low"]), float(row["close"])
        if is_long:
            if lo <= signal.stop_price:
                return _close(signal, ts, signal.stop_price, "STOP")
            if hi >= signal.target_price:
                return _close(signal, ts, signal.target_price, "TARGET")
        else:
            if hi >= signal.stop_price:
                return _close(signal, ts, signal.stop_price, "STOP")
            if lo <= signal.target_price:
                return _close(signal, ts, signal.target_price, "TARGET")
        if ts.time() >= exit_time:
            return _close(signal, ts, close, "TIME")
    return _close(signal, last_ts, last_close, "EOD")


def _close(signal: JudasSignal, ts, price: float, reason: str) -> JudasTrade:
    pnl = (price - signal.entry_price) if signal.direction == LONG else (signal.entry_price - price)
    return JudasTrade(signal=signal, exit_time=pd.Timestamp(ts).to_pydatetime(), exit_price=float(price),
                      exit_reason=reason, pnl_points=float(pnl))


def compute_stats(trades: pd.DataFrame, trading_days: int = 0) -> Dict[str, object]:
    empty = {
        "trading_days": trading_days, "trades": 0, "long_trades": 0, "short_trades": 0,
        "wins": 0, "losses": 0, "win_rate_pct": 0.0, "total_points": 0.0, "avg_points": 0.0,
        "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0, "max_drawdown_points": 0.0,
        "exit_reasons": {},
    }
    if trades is None or trades.empty:
        return empty
    pnl = trades["pnl_points"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    equity = pnl.cumsum()
    drawdown = equity.cummax() - equity
    return {
        "trading_days": trading_days,
        "trades": int(len(pnl)),
        "long_trades": int((trades["direction"] == LONG).sum()),
        "short_trades": int((trades["direction"] == SHORT).sum()),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": round(100.0 * len(wins) / len(pnl), 2),
        "total_points": round(float(pnl.sum()), 2),
        "avg_points": round(float(pnl.mean()), 2),
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "max_drawdown_points": round(float(drawdown.max()), 2),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
    }


class JudasSwingBacktester:
    """
    Date-range backtest of the Judas Swing strategy.

    Pass a ``dhanhq`` client to pull candles from Dhan, or use
    :meth:`run_on_candles` with your own DataFrame / CSV (no credentials needed).
    """

    def __init__(self, dhan=None, security_id: str = NIFTY_50["security_id"],
                 exchange_segment: str = NIFTY_50["exchange_segment"],
                 instrument_type: str = NIFTY_50["instrument_type"],
                 config: Optional[JudasSwingConfig] = None, epoch_is_ist: Optional[bool] = None):
        self.dhan = dhan
        self.security_id = str(security_id)
        self.exchange_segment = exchange_segment
        self.instrument_type = instrument_type
        self.config = config or JudasSwingConfig()
        self.epoch_is_ist = epoch_is_ist
        self.indicator = JudasSwingIndicator(self.config)

    def fetch(self, from_date, to_date, lookback_days: int = 7) -> pd.DataFrame:
        """Download candles, starting ``lookback_days`` earlier so day one has PDH / PDL."""
        if self.dhan is None:
            raise ValueError("A dhanhq client is required to fetch candles; or use run_on_candles()")
        start = pd.Timestamp(from_date).date() - timedelta(days=lookback_days)
        return fetch_intraday_candles(self.dhan, self.security_id, self.exchange_segment, self.instrument_type,
                                      start, to_date, interval=self.config.candle_minutes,
                                      epoch_is_ist=self.epoch_is_ist)

    def run(self, from_date, to_date) -> BacktestResult:
        candles = self.fetch(from_date, to_date)
        return self.run_on_candles(candles, from_date, to_date)

    def run_on_candles(self, candles: pd.DataFrame, from_date=None, to_date=None) -> BacktestResult:
        df = self.indicator.prepare(candles)
        if df.empty:
            raise ValueError("No candles to backtest")
        start = pd.Timestamp(from_date).date() if from_date else df.index[0].date()
        end = pd.Timestamp(to_date).date() if to_date else df.index[-1].date()

        signals = [s for s in self.indicator.signals(df) if start <= s.trade_date <= end]
        by_day = {day: day_df for day, day_df in df.groupby(df.index.date)}
        trades = [simulate_trade(s, by_day[s.trade_date], self.config.exit_time) for s in signals]
        rows = [t.to_row() for t in trades]
        trades_df = pd.DataFrame(rows, columns=TRADE_COLUMNS)
        if not trades_df.empty:
            trades_df["cum_points"] = trades_df["pnl_points"].cumsum().round(2)
        trading_days = sum(1 for d in by_day if start <= d <= end)
        in_range = df[(df.index.date >= start) & (df.index.date <= end)]
        indicator_df = self.indicator.compute(df)
        indicator_df = indicator_df.loc[in_range.index] if not in_range.empty else indicator_df
        return BacktestResult(config=self.config, from_date=start, to_date=end, trades=trades_df,
                              signals=signals, trading_days=trading_days, indicator=indicator_df)
