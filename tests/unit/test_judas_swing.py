from datetime import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from dhanhq.strategies import (
    JudasSwingBacktester,
    JudasSwingConfig,
    JudasSwingIndicator,
    fetch_intraday_candles,
    normalize_candles,
    resample_candles,
)
from dhanhq.strategies.judas_swing import LONG, SHORT, compute_stats

IST = "Asia/Kolkata"


def make_day(day, bars, pad_to=75, pad_bar=None):
    """Build a 5-min session starting 09:15 from (o, h, l, c) tuples, padded to a full day."""
    bars = list(bars)
    if pad_bar is None:
        pad_bar = bars[-1]
    bars += [pad_bar] * (pad_to - len(bars))
    idx = pd.date_range(f"{day} 09:15", periods=len(bars), freq="5min", tz=IST)
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 0
    return df


FLAT_DAY = [(100, 101, 99, 100)]  # PDH = 101, PDL = 99


def long_setup_day():
    # open 100, 09:20 sweeps PDL (low 97), 09:25 green reversal closes 100 > 99,
    # entry 09:30 @ 100.5, 09:35 rallies through the 40pt target.
    return [
        (100, 100.5, 99.5, 99.6),
        (99.6, 99.8, 97, 98),
        (98, 100.2, 97.8, 100),
        (100.5, 101, 100, 100.8),
        (100.8, 160, 100.5, 150),
        (150, 151, 149, 150),
    ]


def short_setup_day():
    return [
        (100, 100.5, 99.5, 100.4),
        (100.4, 103, 100.2, 102),      # sweeps PDH 101
        (102, 102.2, 99.8, 100),       # red reversal closes 100 < 101
        (99.5, 100, 99, 99.2),         # entry 09:30 @ 99.5
        (99.2, 99.5, 40, 50),          # target hit
        (50, 51, 49, 50),
    ]


class TestNormalize:
    def test_columnar_dhan_payload_epoch_utc(self):
        # 2025-09-02 09:15 IST == 03:45 UTC
        ts0 = int(pd.Timestamp("2025-09-02 09:15", tz=IST).timestamp())
        data = {
            "open": [1, 2], "high": [2, 3], "low": [0, 1], "close": [1, 2], "volume": [0, 0],
            "timestamp": [ts0, ts0 + 300],
        }
        df = normalize_candles({"status": "success", "data": data})
        assert list(df.index.time) == [time(9, 15), time(9, 20)]
        assert str(df.index.tz) == IST

    def test_epoch_encoded_as_ist_wallclock_is_autodetected(self):
        # Some Dhan endpoints return IST wall-clock seconds "as if UTC".
        naive = pd.Timestamp("2025-09-02 09:15")
        ts0 = int((naive - pd.Timestamp("1970-01-01")).total_seconds())
        data = {"open": [1], "high": [2], "low": [0], "close": [1], "timestamp": [ts0]}
        df = normalize_candles(data)
        assert df.index[0].time() == time(9, 15)

    def test_string_timestamps_and_missing_volume(self):
        data = {"timestamp": ["2025-09-02 09:15:00", "2025-09-02 09:20:00"],
                "open": [1, 2], "high": [2, 3], "low": [0, 1], "close": [1, 2]}
        df = normalize_candles(data)
        assert "volume" in df.columns
        assert len(df) == 2

    def test_missing_ohlc_raises(self):
        with pytest.raises(ValueError):
            normalize_candles({"timestamp": [1], "open": [1], "high": [1], "low": [1]})

    def test_resample_1min_to_5min(self):
        idx = pd.date_range("2025-09-02 09:15", periods=10, freq="1min", tz=IST)
        df = pd.DataFrame({"open": range(10), "high": range(10, 20), "low": range(0, 10),
                           "close": range(5, 15), "volume": 1}, index=idx)
        out = resample_candles(df, 5)
        assert list(out.index.time) == [time(9, 15), time(9, 20)]
        assert out["open"].iloc[0] == 0 and out["close"].iloc[0] == 9
        assert out["high"].iloc[1] == 19 and out["volume"].iloc[1] == 5


class TestIndicator:
    def test_bullish_judas_swing_detected(self):
        df = pd.concat([make_day("2025-09-01", FLAT_DAY), make_day("2025-09-02", long_setup_day())])
        sigs = JudasSwingIndicator().signals(df)
        assert len(sigs) == 1
        s = sigs[0]
        assert s.direction == LONG
        assert s.swept_level_name == "PDL" and s.swept_level == 99
        assert s.sweep_time.time() == time(9, 20)
        assert s.reversal_time.time() == time(9, 25)
        assert s.entry_time.time() == time(9, 30)
        assert s.entry_price == 100.5
        assert s.stop_price == 97 - 5
        assert s.target_price == 100.5 + 40

    def test_bearish_judas_swing_detected(self):
        df = pd.concat([make_day("2025-09-01", FLAT_DAY), make_day("2025-09-02", short_setup_day())])
        sigs = JudasSwingIndicator().signals(df)
        assert len(sigs) == 1
        s = sigs[0]
        assert s.direction == SHORT and s.swept_level_name == "PDH"
        assert s.entry_price == 99.5 and s.stop_price == 103 + 5

    def test_direction_filter(self):
        df = pd.concat([make_day("2025-09-01", FLAT_DAY), make_day("2025-09-02", short_setup_day())])
        assert JudasSwingIndicator(JudasSwingConfig(direction="long")).signals(df) == []

    def test_no_sweep_no_signal(self):
        df = pd.concat([make_day("2025-09-01", FLAT_DAY), make_day("2025-09-02", [(100, 100.8, 99.2, 100)])])
        assert JudasSwingIndicator().signals(df) == []

    def test_sweep_outside_window_is_ignored(self):
        # Push the sweep to 09:50 (after the 30-min window) -> no signal.
        bars = [(100, 100.5, 99.5, 100)] * 7 + long_setup_day()[1:]
        df = pd.concat([make_day("2025-09-01", FLAT_DAY), make_day("2025-09-02", bars)])
        assert JudasSwingIndicator().signals(df) == []
        assert len(JudasSwingIndicator(JudasSwingConfig(judas_window_minutes=45)).signals(df)) == 1

    def test_first_day_uses_opening_range_only(self):
        # No previous day -> PDL unknown. Opening range (15 min) low = 97.5; sweep it at 09:30.
        bars = [
            (100, 100.5, 98, 99), (99, 99.5, 97.5, 98), (98, 98.5, 97.6, 98),   # OR 09:15-09:30, ORL 97.5
            (98, 98.2, 96, 97),                                                 # 09:30 sweeps ORL
            (97, 99, 96.8, 98.5),                                               # green close > 97.5
            (98.7, 99, 98, 98.8),                                               # entry 09:40
        ]
        df = make_day("2025-09-02", bars)
        sigs = JudasSwingIndicator().signals(df)
        assert len(sigs) == 1
        assert sigs[0].swept_level_name == "ORL" and sigs[0].swept_level == 97.5
        assert sigs[0].entry_time.time() == time(9, 40)

    def test_max_stop_filter_skips_wide_risk(self):
        df = pd.concat([make_day("2025-09-01", FLAT_DAY), make_day("2025-09-02", long_setup_day())])
        assert JudasSwingIndicator(JudasSwingConfig(max_stop_points=5)).signals(df) == []

    def test_compute_frame_has_levels_and_flags(self):
        df = pd.concat([make_day("2025-09-01", FLAT_DAY), make_day("2025-09-02", long_setup_day())])
        ind = JudasSwingIndicator().compute(df)
        d2 = ind.loc["2025-09-02"]
        assert d2["pdh"].iloc[0] == 101 and d2["pdl"].iloc[0] == 99
        assert pd.isna(d2["orl"].iloc[0]) and d2["orl"].iloc[3] == 97   # OR known from 09:30
        assert bool(d2["sweep_low"].iloc[1]) and bool(d2["long_entry"].iloc[3])
        assert d2["signal"].iloc[3] == "LONG:PDL"

    def test_config_validation(self):
        with pytest.raises(ValueError):
            JudasSwingConfig(direction="up")
        with pytest.raises(ValueError):
            JudasSwingConfig(levels=("PDL", "XYZ"))


class TestBacktester:
    def _df(self, bars):
        return pd.concat([make_day("2025-09-01", FLAT_DAY), make_day("2025-09-02", bars)])

    def test_target_hit(self):
        res = JudasSwingBacktester().run_on_candles(self._df(long_setup_day()))
        assert len(res.trades) == 1
        row = res.trades.iloc[0]
        assert row["exit_reason"] == "TARGET" and row["pnl_points"] == 40
        assert res.trading_days == 2
        assert res.stats()["win_rate_pct"] == 100

    def test_stop_hit_and_conservative_same_candle(self):
        bars = long_setup_day()[:4] + [(100.8, 200, 50, 150)]  # both stop & target in one candle
        res = JudasSwingBacktester().run_on_candles(self._df(bars))
        row = res.trades.iloc[0]
        assert row["exit_reason"] == "STOP"
        assert row["pnl_points"] == pytest.approx(92 - 100.5)

    def test_time_exit(self):
        bars = long_setup_day()[:4] + [(100.8, 101, 100.5, 100.9)]
        res = JudasSwingBacktester(config=JudasSwingConfig(exit_time=time(15, 15))).run_on_candles(self._df(bars))
        row = res.trades.iloc[0]
        assert row["exit_reason"] == "TIME"
        assert pd.Timestamp(row["exit_time"]).time() == time(15, 15)
        assert row["pnl_points"] == pytest.approx(100.9 - 100.5)

    def test_date_filter_excludes_days(self):
        res = JudasSwingBacktester().run_on_candles(self._df(long_setup_day()), "2025-09-01", "2025-09-01")
        assert res.trades.empty and res.trading_days == 1
        assert res.summary().startswith("Judas Swing backtest 2025-09-01 -> 2025-09-01")

    def test_run_fetches_with_lookback_and_chunks(self):
        dhan = MagicMock()
        ts0 = int(pd.Timestamp("2025-09-02 09:15", tz=IST).timestamp())
        dhan.intraday_minute_data.return_value = {
            "status": "success",
            "data": {"open": [100], "high": [101], "low": [99], "close": [100], "volume": [0], "timestamp": [ts0]},
        }
        bt = JudasSwingBacktester(dhan)
        bt.fetch("2025-09-02", "2025-09-02", lookback_days=7)
        kwargs = dhan.intraday_minute_data.call_args.kwargs
        assert kwargs["security_id"] == "13" and kwargs["exchange_segment"] == "IDX_I"
        assert kwargs["instrument_type"] == "INDEX" and kwargs["interval"] == 5
        assert kwargs["from_date"] == "2025-08-26" and kwargs["to_date"] == "2025-09-02"

        dhan.intraday_minute_data.reset_mock()
        fetch_intraday_candles(dhan, "13", "IDX_I", "INDEX", "2025-01-01", "2025-05-30", chunk_days=75)
        calls = [c.kwargs for c in dhan.intraday_minute_data.call_args_list]
        assert [(c["from_date"], c["to_date"]) for c in calls] == [
            ("2025-01-01", "2025-03-16"), ("2025-03-17", "2025-05-30")]

    def test_fetch_failure_raises(self):
        dhan = MagicMock()
        dhan.intraday_minute_data.return_value = {"status": "failure", "remarks": "bad token"}
        with pytest.raises(RuntimeError):
            fetch_intraday_candles(dhan, "13", "IDX_I", "INDEX", "2025-09-01", "2025-09-02")

    def test_stats_on_empty(self):
        s = compute_stats(pd.DataFrame())
        assert s["trades"] == 0 and s["profit_factor"] == 0.0
