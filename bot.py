"""
AdvancedRSI Backtesting Engine — Professional Edition
======================================================
Features:
  - RSI × WMA crossover strategy with ATR trailing stop + partial exit
  - 0.4% expense per side (brokerage + STT approximation)
  - FY-split P/L buckets (Indian FY: Apr 1 → Mar 31)
  - Full tax-ready report: gross wins, gross losses, expenses, STCG estimate
  - Avg win / avg loss, realised R:R, profit factor, expectancy
  - Telegram report with all hedge-fund-grade metrics
"""

import backtrader as bt
import yfinance as yf
import pandas as pd
import os
import requests
import time
import collections
import collections.abc
import numpy as np
from datetime import datetime

# ── Compatibility patch ──────────────────────────────────────────────────────
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    # Telegram — move these to env vars in production
    "TOKEN":          os.getenv("TG_TOKEN",   "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg"),
    "CHAT_ID":        os.getenv("TG_CHAT_ID", "1950462171"),

    "INITIAL_CASH":   50_000,
    "RISK_PERCENT":   0.02,          # 2 % of portfolio per trade
    "EXPENSE_RATE":   0.004,         # 0.4 % per side (buy + sell)
    "BACKTEST_PERIOD":"2y",
    "WATCHLIST_FILE": "watchlist.txt",
    "TEST_TIMEFRAMES":["2h"],
    "DELAY_SECONDS":  5,
}

# Indian Financial Year helper
def get_fy(dt: datetime) -> str:
    """Return 'FY2425' style label for a datetime."""
    if dt.month >= 4:
        return f"FY{str(dt.year)[2:]}{str(dt.year + 1)[2:]}"
    return f"FY{str(dt.year - 1)[2:]}{str(dt.year)[2:]}"


# ── Trade observer — captures every closed trade with full detail ─────────────
class TradeLogger(bt.Analyzer):
    """Stores every completed trade as a dict for downstream processing."""

    def start(self):
        self.trades = []

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        dt = self.strategy.data.datetime.datetime(0)
        gross_pnl = trade.pnl           # before commission
        commission = trade.commission   # backtrader commission (set to 0 here)
        # Manual expense: 0.4% on entry value + 0.4% on exit value
        entry_val = abs(trade.price * trade.size)
        exit_val  = abs(self.strategy.data.close[0] * trade.size)
        expense   = (entry_val + exit_val) * CONFIG["EXPENSE_RATE"]
        net_pnl   = gross_pnl - expense

        self.trades.append({
            "dt":        dt,
            "fy":        get_fy(dt),
            "ticker":    self.strategy.data._name,
            "size":      trade.size,
            "entry":     trade.price,
            "gross_pnl": gross_pnl,
            "expense":   expense,
            "net_pnl":   net_pnl,
            "won":       net_pnl > 0,
        })

    def get_analysis(self):
        return self.trades


# ── Strategy ─────────────────────────────────────────────────────────────────
class AdvancedRSIStrategy(bt.Strategy):
    params = (
        ("rsi_p",  40),
        ("wma_p",  15),
        ("atr_p",  14),
        ("rr",     2.5),
        ("atr_m",  2.0),
    )

    def __init__(self):
        self.rsi     = bt.indicators.RSI(self.data.close, period=self.p.rsi_p)
        self.wma_rsi = bt.indicators.WMA(self.rsi,        period=self.p.wma_p)
        self.atr     = bt.indicators.ATR(self.data,       period=self.p.atr_p)
        self.stop_loss   = None
        self.target      = None
        self.half_booked = False

    def next(self):
        if not self.position:
            # Entry: RSI crosses above its WMA
            if self.rsi[0] > self.wma_rsi[0] and self.rsi[-1] <= self.wma_rsi[-1]:
                entry = self.data.close[0]
                risk  = entry - self.data.low[-1]
                if risk > 0:
                    qty       = int((self.broker.get_value() * CONFIG["RISK_PERCENT"]) / risk)
                    final_qty = min(qty, int(self.broker.get_cash() / entry))
                    if final_qty > 0:
                        self.buy(size=final_qty)
                        self.stop_loss   = self.data.low[-1]
                        self.target      = entry + (risk * self.p.rr)
                        self.half_booked = False

        elif self.position:
            # Partial exit at target
            if not self.half_booked and self.data.high[0] >= self.target:
                self.sell(size=int(self.position.size / 2))
                self.half_booked = True
                self.stop_loss   = max(self.stop_loss, self.data.open[0])

            # ATR trailing stop after partial exit
            if self.half_booked:
                self.stop_loss = max(
                    self.stop_loss,
                    self.data.close[0] - (self.atr[0] * self.p.atr_m),
                )

            # Stop-loss hit
            if self.data.low[0] <= self.stop_loss:
                self.close()


# ── Helpers ───────────────────────────────────────────────────────────────────
def send_msg(text: str):
    url = f"https://api.telegram.org/bot{CONFIG['TOKEN']}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": CONFIG["CHAT_ID"], "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"[Telegram] Failed: {e}")


def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample a 1h OHLCV DataFrame to a coarser timeframe."""
    if timeframe == "1h":
        return df
    tf_map = {"1d": "D", "4h": "4h", "2h": "2h", "1w": "W"}
    rule = tf_map.get(timeframe, timeframe)
    agg  = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    resampled = df.resample(rule).apply(agg).dropna()
    # Drop rows with zero volume (market closed periods after resample)
    resampled = resampled[resampled["Volume"] > 0]
    return resampled


def build_fy_summary(all_trades: list) -> dict:
    """
    Group trades by Indian FY.
    Returns { 'FY2425': { wins, losses, gross_p, gross_l, expenses, net_pnl }, ... }
    """
    fy_data = {}
    for t in all_trades:
        fy = t["fy"]
        if fy not in fy_data:
            fy_data[fy] = {
                "wins": 0, "losses": 0,
                "gross_p": 0.0, "gross_l": 0.0,
                "expenses": 0.0, "net_pnl": 0.0,
            }
        bucket = fy_data[fy]
        bucket["expenses"] += t["expense"]
        if t["won"]:
            bucket["wins"]    += 1
            bucket["gross_p"] += t["gross_pnl"]
        else:
            bucket["losses"]  += 1
            bucket["gross_l"] += abs(t["gross_pnl"])
        bucket["net_pnl"] += t["net_pnl"]
    return fy_data


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Load watchlist ──
    with open(CONFIG["WATCHLIST_FILE"], "r") as f:
        symbols = [s.strip().upper() for s in f.read().splitlines() if s.strip()]

    # ── Aggregation containers ──
    all_trades = []          # every TradeLogger trade dict
    tf_stats   = {
        tf: {"profit": 0.0, "trades": 0, "wins": 0}
        for tf in CONFIG["TEST_TIMEFRAMES"]
    }
    best = {"name": "", "profit": -999_999}
    max_dd_global = 0.0

    # ── Run cerebro for each symbol × timeframe ──
    for symbol in symbols:
        ticker  = symbol if "." in symbol else f"{symbol}.NS"
        raw_df  = yf.download(ticker, period=CONFIG["BACKTEST_PERIOD"],
                              interval="1h", progress=False)
        if raw_df.empty:
            print(f"[SKIP] No data for {ticker}")
            continue

        # Flatten MultiIndex columns from yfinance
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        raw_df.columns = [str(c) for c in raw_df.columns]

        # Need at least 80 bars for indicators to warm up
        if len(raw_df) < 80:
            print(f"[SKIP] Too few bars for {ticker}")
            continue

        for tf in CONFIG["TEST_TIMEFRAMES"]:
            print(f"  Analyzing {ticker} @ {tf} ...")
            df = resample_data(raw_df.copy(), tf)

            if len(df) < 60:
                print(f"  [SKIP] Not enough bars after resample ({len(df)})")
                continue

            cerebro = bt.Cerebro()
            cerebro.adddata(bt.feeds.PandasData(dataname=df), name=f"{ticker}_{tf}")
            cerebro.addstrategy(AdvancedRSIStrategy)
            cerebro.broker.setcash(CONFIG["INITIAL_CASH"])
            cerebro.broker.setcommission(commission=0.0)  # expenses handled manually
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tr")
            cerebro.addanalyzer(bt.analyzers.DrawDown,      _name="dd")
            cerebro.addanalyzer(TradeLogger,                _name="tl")

            try:
                results  = cerebro.run()
                strat    = results[0]
                tr       = strat.analyzers.tr.get_analysis()
                dd       = strat.analyzers.dd.get_analysis()
                trades   = strat.analyzers.tl.get_analysis()

                if not trades:
                    continue

                # Collect per-trade data
                all_trades.extend(trades)

                symbol_wins   = sum(1 for t in trades if t["won"])
                symbol_net    = sum(t["net_pnl"] for t in trades)

                # Timeframe stats
                tf_stats[tf]["profit"] += symbol_net
                tf_stats[tf]["trades"] += len(trades)
                tf_stats[tf]["wins"]   += symbol_wins

                # Global max drawdown
                if dd.get("max") and dd["max"].get("drawdown"):
                    max_dd_global = max(max_dd_global, dd["max"]["drawdown"])

                # Best symbol
                if symbol_net > best["profit"]:
                    best = {"name": f"{ticker} ({tf})", "profit": symbol_net}

            except Exception as e:
                print(f"  [ERROR] {ticker} @ {tf}: {e}")

        time.sleep(CONFIG["DELAY_SECONDS"])

    # ─────────────────────────────────────────────────────────────────────────
    # FINANCIAL CALCULATIONS
    # ─────────────────────────────────────────────────────────────────────────
    if not all_trades:
        send_msg("⚠️ No trades executed. Check watchlist or data.")
        print("No trades found.")
        exit()

    total_trades  = len(all_trades)
    total_wins    = sum(1 for t in all_trades if t["won"])
    total_losses  = total_trades - total_wins

    gross_p       = sum(t["gross_pnl"] for t in all_trades if t["won"])
    gross_l       = sum(abs(t["gross_pnl"]) for t in all_trades if not t["won"])
    total_expenses= sum(t["expense"] for t in all_trades)
    gross_pnl     = gross_p - gross_l
    net_pnl       = gross_pnl - total_expenses

    win_rate      = (total_wins / total_trades * 100) if total_trades > 0 else 0
    profit_factor = (gross_p / gross_l)               if gross_l > 0 else float("inf")
    expectancy    = (net_pnl / total_trades)           if total_trades > 0 else 0
    avg_win       = (gross_p / total_wins)             if total_wins > 0 else 0
    avg_loss      = (gross_l / total_losses)           if total_losses > 0 else 0
    rr_realised   = (avg_win / avg_loss)               if avg_loss > 0 else 0
    roi_pct       = (net_pnl / CONFIG["INITIAL_CASH"]) * 100

    # ── FY-split tax buckets ──
    fy_summary = build_fy_summary(all_trades)
    fy_tax_lines = ""
    total_stcg_tax = 0.0
    for fy, d in sorted(fy_summary.items()):
        # Net the wins vs losses within this FY before applying tax
        fy_net = d["gross_p"] - d["gross_l"] - d["expenses"]
        stcg   = max(0.0, fy_net * 0.15)   # 15% STCG; losses set off same FY
        total_stcg_tax += stcg
        fy_acc = (d["wins"] / (d["wins"] + d["losses"]) * 100) if (d["wins"] + d["losses"]) > 0 else 0
        fy_tax_lines += (
            f"  *{fy}*\n"
            f"    Wins: ₹{d['gross_p']:,.0f}  |  Losses: −₹{d['gross_l']:,.0f}\n"
            f"    Expenses: −₹{d['expenses']:,.0f}  |  Net: ₹{fy_net:,.0f}\n"
            f"    STCG tax (15%): ₹{stcg:,.0f}  |  Accuracy: {fy_acc:.1f}%\n\n"
        )

    net_after_tax = net_pnl - total_stcg_tax

    # ── Timeframe breakdown ──
    tf_lines = ""
    for tf, d in tf_stats.items():
        tf_acc = (d["wins"] / d["trades"] * 100) if d["trades"] > 0 else 0
        tf_lines += (
            f"  *{tf}*: ₹{d['profit']:,.0f}  |  Acc: {tf_acc:.1f}%  |  Trades: {d['trades']}\n"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # MASTER REPORT
    # ─────────────────────────────────────────────────────────────────────────
    report = (
        f"🏛 *FINANCIAL PERFORMANCE AUDIT*\n"
        f"Period: {CONFIG['BACKTEST_PERIOD']}  |  Capital: ₹{CONFIG['INITIAL_CASH']:,}\n"
        f"Strategy: AdvancedRSI × WMA  |  Expense: 0.4%/side\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"

        f"*── 2-YEAR P/L BREAKDOWN ──*\n"
        f"  Gross winning trades : +₹{gross_p:>12,.2f}\n"
        f"  Gross losing trades  : −₹{gross_l:>12,.2f}\n"
        f"  Gross P/L            :  ₹{gross_pnl:>12,.2f}\n"
        f"  Total expenses       : −₹{total_expenses:>12,.2f}\n"
        f"  *Net realised P/L*   :  *₹{net_pnl:>10,.2f}*\n"
        f"  STCG tax est. (15%)  : −₹{total_stcg_tax:>12,.2f}\n"
        f"  *Net after tax*      :  *₹{net_after_tax:>10,.2f}*\n"
        f"  Return on capital    :  {roi_pct:.2f}%\n\n"

        f"*── STRATEGY METRICS ──*\n"
        f"  Trades     : {total_trades}  (Wins: {total_wins}  Losses: {total_losses})\n"
        f"  Accuracy   : {win_rate:.1f}%\n"
        f"  Profit Factor: {profit_factor:.2f}\n"
        f"  Expectancy : ₹{expectancy:.0f} / trade\n"
        f"  Avg Win    : ₹{avg_win:.0f}  |  Avg Loss: ₹{avg_loss:.0f}\n"
        f"  Realised R:R: {rr_realised:.2f}x\n"
        f"  Max Drawdown: {max_dd_global:.2f}%\n\n"

        f"*── FY-WISE TAX BUCKETS ──*\n"
        f"{fy_tax_lines}"

        f"*── TIMEFRAME BREAKDOWN ──*\n"
        f"{tf_lines}\n"

        f"*── STAR PERFORMER ──*\n"
        f"  {best['name']}  (+₹{best['profit']:,.0f})\n\n"

        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"_Losses set off vs gains within same FY (Indian IT rules)._\n"
        f"_STCG @ 15% flat. Carry-forward losses valid 8 yrs._\n"
        f"_Consult your CA before filing. Auto-generated report._"
    )

    print("\n" + report)
    send_msg(report)
    print("\n[Done] Report sent to Telegram.")
