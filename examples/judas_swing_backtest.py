"""
ICT Judas Swing (Morning Trap) - Nifty backtest by date range.

Usage (credentials from env CLIENT_ID / ACCESS_TOKEN, or --client-id / --access-token):

    python examples/judas_swing_backtest.py --from-date 2025-08-01 --to-date 2025-09-30

    # Bank Nifty, long-only, 50 point target, 09:15-09:45 window
    python examples/judas_swing_backtest.py --from-date 2025-09-01 --to-date 2025-09-30 \
        --symbol BANKNIFTY --direction long --target 50 --window 30

    # Offline: backtest a CSV of 5-min (or 1-min) candles - no API needed
    python examples/judas_swing_backtest.py --csv nifty_5min.csv --from-date 2025-09-01 --to-date 2025-09-30

The CSV needs open, high, low, close columns and a timestamp column
(epoch seconds like Dhan, or "YYYY-MM-DD HH:MM:SS" IST).
"""
import argparse
import os
import sys

import pandas as pd

from dhanhq.strategies import (
    BANK_NIFTY,
    NIFTY_50,
    JudasSwingBacktester,
    JudasSwingConfig,
    normalize_candles,
)

SYMBOLS = {"NIFTY": NIFTY_50, "BANKNIFTY": BANK_NIFTY}


def parse_args():
    p = argparse.ArgumentParser(description="ICT Judas Swing backtest (Dhan 5-min candles)")
    p.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--symbol", default="NIFTY", choices=sorted(SYMBOLS), help="Preset index (default NIFTY)")
    p.add_argument("--security-id", help="Override Dhan security id (e.g. 13 = NIFTY 50)")
    p.add_argument("--segment", help="Override exchange segment (IDX_I, NSE_EQ, NSE_FNO ...)")
    p.add_argument("--instrument", help="Override instrument type (INDEX, EQUITY, FUTIDX ...)")
    p.add_argument("--csv", help="Backtest this candle CSV instead of calling the Dhan API")
    p.add_argument("--client-id", default=os.getenv("CLIENT_ID"))
    p.add_argument("--access-token", default=os.getenv("ACCESS_TOKEN"))
    p.add_argument("--direction", default="both", choices=["long", "short", "both"])
    p.add_argument("--target", type=float, default=40.0, help="Target in points (Nifty 30-50)")
    p.add_argument("--sl-buffer", type=float, default=5.0, help="Stop buffer beyond the sweep extreme")
    p.add_argument("--max-stop", type=float, default=60.0, help="Skip trades riskier than this (0 = no limit)")
    p.add_argument("--window", type=int, default=30, help="Judas window minutes after 09:15 (15-30)")
    p.add_argument("--opening-range", type=int, default=15, help="Opening range minutes")
    p.add_argument("--entry-cutoff", default="10:15", help="No entries after HH:MM")
    p.add_argument("--exit-time", default="15:15", help="Time square-off HH:MM")
    p.add_argument("--out", help="Write the trade log to this CSV")
    p.add_argument("--indicator-out", help="Write the per-candle indicator frame to this CSV")
    return p.parse_args()


def hhmm(value):
    hh, mm = value.split(":")
    from datetime import time
    return time(int(hh), int(mm))


def main():
    args = parse_args()
    config = JudasSwingConfig(
        direction=args.direction,
        target_points=args.target,
        stop_buffer_points=args.sl_buffer,
        max_stop_points=args.max_stop or None,
        judas_window_minutes=args.window,
        opening_range_minutes=args.opening_range,
        entry_cutoff=hhmm(args.entry_cutoff),
        exit_time=hhmm(args.exit_time),
    )
    instrument = dict(SYMBOLS[args.symbol])
    if args.security_id:
        instrument["security_id"] = args.security_id
    if args.segment:
        instrument["exchange_segment"] = args.segment
    if args.instrument:
        instrument["instrument_type"] = args.instrument

    if args.csv:
        candles = normalize_candles(pd.read_csv(args.csv))
        bt = JudasSwingBacktester(config=config, **instrument)
        result = bt.run_on_candles(candles, args.from_date, args.to_date)
    else:
        if not args.client_id or not args.access_token:
            sys.exit("Need --client-id/--access-token (or CLIENT_ID / ACCESS_TOKEN env vars), or use --csv")
        from dhanhq import DhanContext, dhanhq
        dhan = dhanhq(DhanContext(args.client_id, args.access_token))
        bt = JudasSwingBacktester(dhan, config=config, **instrument)
        result = bt.run(args.from_date, args.to_date)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(result.summary())
    print()
    cols = ["date", "direction", "swept_level", "level_price", "entry_time", "entry", "stop", "target",
            "exit_time", "exit", "exit_reason", "pnl_points", "cum_points"]
    if result.trades.empty:
        print("No Judas Swing setups found in this range.")
    else:
        print(result.trades[cols].to_string(index=False))

    if args.out:
        result.trades.to_csv(args.out, index=False)
        print(f"\nTrade log written to {args.out}")
    if args.indicator_out and result.indicator is not None:
        result.indicator.to_csv(args.indicator_out)
        print(f"Indicator frame written to {args.indicator_out}")


if __name__ == "__main__":
    main()
