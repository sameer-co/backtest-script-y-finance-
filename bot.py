#!/usr/bin/env python3
"""
EMA(9) / SMA(6)-of-EMA Crossover Strategy — Backtester (Binance public data)

Replicates the Pine Script strategy exactly:
  - Main line:  EMA(length=9) on close
  - Smoothing:  SMA(length=6) applied ON THE EMA VALUES (not on price)
  - Entry:      Long when EMA crosses ABOVE the SMA-of-EMA
  - Stop Loss:  entry_price - sl_mult * ATR(atr_len)          [default sl_mult=1.5]
  - Take Profit: entry_price + rr_mult * (entry_price - SL)   [default rr_mult=3.0]
  - Long only, one position at a time

Data source: Binance public REST API (no API key / auth needed)
  GET https://api.binance.com/api/v3/klines

USAGE
-----
Single backtest with default params (9 EMA / 6 SMA / ATR14 / 1.5x SL / 3x TP):
    python backtest_ema_sma_atr.py --symbol BTCUSDT --interval 1h --start 2023-01-01 --end 2024-01-01

Custom parameters:
    python backtest_ema_sma_atr.py --symbol ETHUSDT --interval 4h --ema_len 9 --sma_len 6 \
        --atr_len 14 --sl_mult 1.5 --rr_mult 3.0

Parameter tuning (grid search over a range of values):
    python backtest_ema_sma_atr.py --symbol BTCUSDT --interval 1h --tune

Outputs:
  <out>_trades.csv          full trade-by-trade log
  <out>_chart.png           price+signals chart and equity curve
  <out>_tuning_results.csv  full grid search results (only with --tune)
"""

import argparse
import time
import itertools
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


# ======================================================================
# DATA FETCHING (Binance public endpoint — no API key required)
# ======================================================================
def fetch_binance_klines(symbol, interval, start_str, end_str):
    """Pull OHLCV candles from Binance's public klines endpoint, paginating past the 1000-candle limit."""
    start_ts = int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str, tz="UTC").timestamp() * 1000)

    all_rows = []
    cur = start_ts
    while cur < end_ts:
        params = {"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ts, "limit": 1000}
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        last_open_time = data[-1][0]
        if last_open_time == cur:
            break
        cur = last_open_time + 1
        if len(data) < 1000:
            break
        time.sleep(0.25)  # stay polite to the public rate limit

    if not all_rows:
        raise RuntimeError("No data returned from Binance. Check symbol / interval / date range.")

    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].set_index("open_time")
    df = df[~df.index.duplicated(keep="first")]
    return df


# ======================================================================
# INDICATORS
# ======================================================================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def sma(series, length):
    return series.rolling(length).mean()


def atr(df, length):
    """Wilder/RMA-smoothed ATR — matches Pine's ta.atr() exactly."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


# ======================================================================
# BACKTEST ENGINE
# ======================================================================
def run_backtest(df, ema_len=9, sma_len=6, atr_len=14, sl_mult=1.5, rr_mult=3.0,
                  initial_capital=100_000.0, qty_pct=1.0, round_trip_fee_pct=0.0):
    """
    Long-only backtest, mirrors the Pine strategy:
      entry on EMA crossing above SMA-of-EMA, ATR-based SL, fixed-RR TP.
    Exit check order within a bar: if both SL and TP could have been touched
    (low <= SL and high >= TP in the same candle), the conservative assumption
    is SL fills first, since intrabar path is unknown from OHLC alone.

    round_trip_fee_pct: TOTAL cost for one full buy->sell cycle, as a fraction
        (e.g. 0.006 = 0.6%). Use this to model a full swap cycle like
        USDC->SOL->USDC on a DEX (swap fee + slippage on both legs combined),
        or a CEX round trip (taker fee in + taker fee out). Charged once per
        trade against the position's entry notional.
    """
    d = df.copy()
    d["ema"] = ema(d["close"], ema_len)
    d["sma_of_ema"] = sma(d["ema"], sma_len)
    d["atr"] = atr(d, atr_len)
    d["cross_up"] = (d["ema"] > d["sma_of_ema"]) & (d["ema"].shift(1) <= d["sma_of_ema"].shift(1))

    equity = initial_capital
    position = None
    trades = []
    equity_curve = []

    for t, row in d.iterrows():
        if position is not None:
            hit_sl = row["low"] <= position["sl"]
            hit_tp = row["high"] >= position["tp"]
            exit_price, reason = None, None
            if hit_sl:
                exit_price, reason = position["sl"], "SL"   # SL wins if both hit same bar
            elif hit_tp:
                exit_price, reason = position["tp"], "TP"

            if exit_price is not None:
                pnl = (exit_price - position["entry_price"]) * position["qty"]
                notional = position["entry_price"] * position["qty"]
                fee = notional * round_trip_fee_pct  # e.g. 0.6% of position size, once per full cycle
                pnl -= fee
                equity += pnl
                trades.append({
                    "entry_time": position["entry_time"], "exit_time": t,
                    "entry_price": position["entry_price"], "exit_price": exit_price,
                    "sl": position["sl"], "tp": position["tp"], "qty": position["qty"],
                    "fee": fee, "pnl": pnl, "pnl_pct": pnl / notional * 100,
                    "reason": reason, "bars_held": position["bars_held"],
                })
                position = None
            else:
                position["bars_held"] += 1

        if position is None and row["cross_up"] and not np.isnan(row["atr"]):
            entry_price = row["close"]
            sl = entry_price - sl_mult * row["atr"]
            risk = entry_price - sl
            if risk > 0:
                tp = entry_price + rr_mult * risk
                qty = (equity * qty_pct) / entry_price
                position = {"entry_time": t, "entry_price": entry_price, "sl": sl, "tp": tp,
                            "qty": qty, "bars_held": 0}

        unrealized = (row["close"] - position["entry_price"]) * position["qty"] if position else 0.0
        equity_curve.append({"time": t, "equity": equity + unrealized})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index("time")
    return trades_df, equity_df, position


# ======================================================================
# STATS
# ======================================================================
def _max_streak(bool_series, value):
    max_run = run = 0
    for v in bool_series:
        run = run + 1 if v == value else 0
        max_run = max(max_run, run)
    return max_run


def compute_stats(trades_df, equity_df, initial_capital):
    stats = {}
    n = len(trades_df)
    stats["Total Trades"] = n
    if n == 0:
        stats["Note"] = "No trades were triggered in this period/parameter set."
        return stats

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    stats["Winning Trades"] = len(wins)
    stats["Losing Trades"] = len(losses)
    stats["Win Rate %"] = round(len(wins) / n * 100, 2)

    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    stats["Gross Profit"] = round(gross_profit, 2)
    stats["Gross Loss"] = round(gross_loss, 2)
    stats["Profit Factor"] = round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf")

    stats["Avg Win"] = round(wins["pnl"].mean(), 2) if len(wins) else 0.0
    stats["Avg Loss"] = round(losses["pnl"].mean(), 2) if len(losses) else 0.0
    stats["Avg Win %"] = round(wins["pnl_pct"].mean(), 2) if len(wins) else 0.0
    stats["Avg Loss %"] = round(losses["pnl_pct"].mean(), 2) if len(losses) else 0.0
    stats["Largest Win"] = round(trades_df["pnl"].max(), 2)
    stats["Largest Loss"] = round(trades_df["pnl"].min(), 2)
    stats["Avg Bars Held"] = round(trades_df["bars_held"].mean(), 1)

    final_equity = equity_df["equity"].iloc[-1]
    net_profit = final_equity - initial_capital
    stats["Net Profit"] = round(net_profit, 2)
    stats["Net Profit %"] = round(net_profit / initial_capital * 100, 2)
    stats["Final Equity"] = round(final_equity, 2)
    stats["Total Fees Paid"] = round(trades_df["fee"].sum(), 2)

    running_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - running_max) / running_max * 100
    stats["Max Drawdown %"] = round(drawdown.min(), 2)

    stats["Sharpe (per-trade)"] = (round(trades_df["pnl_pct"].mean() / trades_df["pnl_pct"].std(), 3)
                                    if trades_df["pnl_pct"].std() > 0 else float("nan"))

    win_bool = trades_df["pnl"].gt(0)
    stats["Max Consecutive Wins"] = _max_streak(win_bool, True)
    stats["Max Consecutive Losses"] = _max_streak(win_bool, False)

    return stats


def print_stats(stats, title="BACKTEST RESULTS"):
    print("\n" + "=" * 62)
    print(title.center(62))
    print("=" * 62)
    for k, v in stats.items():
        print(f"{k:<25}: {v}")
    print("=" * 62 + "\n")


# ======================================================================
# CHARTING
# ======================================================================
def plot_results(df, equity_df, trades_df, out_prefix="backtest"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(df.index, df["close"], label="Close", color="black", linewidth=0.8)
    axes[0].plot(df.index, df["ema"], label="EMA", color="blue", linewidth=1)
    axes[0].plot(df.index, df["sma_of_ema"], label="SMA (of EMA)", color="orange", linewidth=1)
    if len(trades_df):
        axes[0].scatter(trades_df["entry_time"], trades_df["entry_price"], marker="^",
                         color="green", s=40, label="Entry", zorder=5)
        axes[0].scatter(trades_df["exit_time"], trades_df["exit_price"], marker="v",
                         color="red", s=40, label="Exit", zorder=5)
    axes[0].set_title("Price with EMA / SMA-of-EMA Signals")
    axes[0].legend(loc="upper left")

    axes[1].plot(equity_df.index, equity_df["equity"], color="purple")
    axes[1].axhline(equity_df["equity"].iloc[0], color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("Equity Curve")

    plt.tight_layout()
    out_path = f"{out_prefix}_chart.png"
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ======================================================================
# PARAMETER TUNING (grid search)
# ======================================================================
def tune_parameters(df, param_grid, initial_capital=100_000.0, round_trip_fee_pct=0.0,
                     metric="Net Profit %", top_n=15):
    """
    param_grid example:
        {"ema_len": [7,9,11], "sma_len": [4,6,8], "atr_len": [14],
         "sl_mult": [1.0,1.5,2.0], "rr_mult": [2.0,3.0,4.0]}
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    print(f"Running grid search over {len(combos)} parameter combinations...")

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        trades_df, equity_df, _ = run_backtest(df, initial_capital=initial_capital,
                                                round_trip_fee_pct=round_trip_fee_pct, **params)
        stats = compute_stats(trades_df, equity_df, initial_capital)
        row = {**params,
               "Total Trades": stats.get("Total Trades", 0),
               "Win Rate %": stats.get("Win Rate %", np.nan),
               "Net Profit %": stats.get("Net Profit %", np.nan),
               "Profit Factor": stats.get("Profit Factor", np.nan),
               "Max Drawdown %": stats.get("Max Drawdown %", np.nan)}
        results.append(row)
        if i % max(1, len(combos) // 10) == 0:
            print(f"  {i}/{len(combos)} done...")

    results_df = pd.DataFrame(results)
    results_df = results_df[results_df["Total Trades"] > 0]
    results_df = results_df.sort_values(metric, ascending=False).reset_index(drop=True)
    return results_df.head(top_n), results_df


# ======================================================================
# CLI
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Backtest EMA/SMA-of-EMA crossover strategy on Binance data")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h", help="1m,5m,15m,1h,4h,1d,...")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--ema_len", type=int, default=9)
    ap.add_argument("--sma_len", type=int, default=6)
    ap.add_argument("--atr_len", type=int, default=14)
    ap.add_argument("--sl_mult", type=float, default=1.5)
    ap.add_argument("--rr_mult", type=float, default=3.0)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--round_trip_fee_pct", type=float, default=0.006,
                     help="TOTAL cost for a full buy->sell cycle as a fraction, e.g. 0.006 = 0.6%% "
                          "(default models a DEX swap round trip like USDC->SOL->USDC, incl. fee+slippage "
                          "on both legs). Use ~0.0008 for a Binance CEX round trip (0.04%% x2). Set 0 to ignore fees.")
    ap.add_argument("--tune", action="store_true", help="run a parameter grid search instead of one backtest")
    ap.add_argument("--out", default="ema_sma_atr")
    args = ap.parse_args()

    print(f"Fetching {args.symbol} {args.interval} data from Binance ({args.start} to {args.end})...")
    df = fetch_binance_klines(args.symbol, args.interval, args.start, args.end)
    print(f"Fetched {len(df)} candles.")

    if args.tune:
        param_grid = {
            "ema_len": [7, 9, 11, 13],
            "sma_len": [4, 6, 8, 10],
            "atr_len": [10, 14, 20],
            "sl_mult": [1.0, 1.5, 2.0],
            "rr_mult": [2.0, 3.0, 4.0],
        }
        top_results, all_results = tune_parameters(df, param_grid, initial_capital=args.capital,
                                                    round_trip_fee_pct=args.round_trip_fee_pct)
        print("\nTop parameter combinations by Net Profit %:\n")
        print(top_results.to_string(index=False))
        all_results.to_csv(f"{args.out}_tuning_results.csv", index=False)
        print(f"\nFull grid search results ({len(all_results)} combos) saved to {args.out}_tuning_results.csv")
        return

    trades_df, equity_df, open_pos = run_backtest(
        df, ema_len=args.ema_len, sma_len=args.sma_len, atr_len=args.atr_len,
        sl_mult=args.sl_mult, rr_mult=args.rr_mult, initial_capital=args.capital,
        round_trip_fee_pct=args.round_trip_fee_pct)

    stats = compute_stats(trades_df, equity_df, args.capital)
    title = f"{args.symbol} {args.interval} | EMA{args.ema_len}/SMA{args.sma_len}-of-EMA | ATR{args.atr_len} SL{args.sl_mult}x TP{args.rr_mult}x"
    print_stats(stats, title=title)

    if open_pos is not None:
        print(f"Note: a position was still open at end of data (entered {open_pos['entry_time']}), not counted above.\n")

    if len(trades_df):
        trades_df.to_csv(f"{args.out}_trades.csv", index=False)
        print(f"Trade log saved to {args.out}_trades.csv")

    df["ema"] = ema(df["close"], args.ema_len)
    df["sma_of_ema"] = sma(df["ema"], args.sma_len)
    chart_path = plot_results(df, equity_df, trades_df, out_prefix=args.out)
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
