"""
AdvancedRSI Backtesting Engine — Professional Edition v3
=========================================================

What's new vs v2:
  - Custom date range: START_DATE / END_DATE (not period string)
    Default: 01-Jun-2019 → 01-Jun-2025 (6-year backtest)
  - Multiple timeframes run independently: 1h, 2h, 4h, 1d
  - Each timeframe gets its OWN full Telegram report
  - Final MASTER SUMMARY compares all timeframes side-by-side
  - Added metrics: Sortino ratio, Win/Loss streak, Avg hold bars,
    Recovery factor, Monthly P&L table, Yearly P&L breakdown
  - Walk-forward per timeframe (75% IS / 25% OOS split)
  - FY-wise tax buckets per timeframe
  - Expense fix: uses trade.value not trade.size
  - EMA200 trend filter + next-candle entry + max 5 positions

Usage:
  export TG_TOKEN="your_bot_token"
  export TG_CHAT_ID="your_chat_id"
  python backtest_pro.py
"""

import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import os, requests, time, collections, collections.abc
from datetime import datetime, date
from collections import defaultdict

# ── Compatibility ─────────────────────────────────────────────────────────────
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "TOKEN":          os.getenv("TG_TOKEN"),
    "CHAT_ID":        os.getenv("TG_CHAT_ID"),

    # Date range — change these to test different windows
    "START_DATE":     "2019-06-01",
    "END_DATE":       "2025-06-01",

    "INITIAL_CASH":   50_000,
    "RISK_PERCENT":   0.02,       # 2% equity risk per trade
    "EXPENSE_RATE":   0.004,      # 0.4% per side (STT + brokerage)
    "SLIPPAGE_PERC":  0.001,      # 0.1% per side
    "MAX_POSITIONS":  5,          # max simultaneous open positions

    "WATCHLIST_FILE": "watchlist.txt",

    # Timeframes to test — each gets a separate full report
    # yfinance supports: "1h", "2h", "4h", "1d"
    # NOTE: 1h data from yfinance is limited to last 730 days regardless of
    # START_DATE. For 6-year tests, use 2h / 4h / 1d only.
    "TEST_TIMEFRAMES": ["2h", "4h", "1d"],

    # Walk-forward split: IS_RATIO of data = in-sample
    "IS_RATIO":       0.75,       # 75% in-sample, 25% out-of-sample

    "DELAY_SECONDS":  3,          # pause between symbols to avoid rate limits
}

if not CONFIG["TOKEN"] or not CONFIG["CHAT_ID"]:
    print("[WARNING] TG_TOKEN / TG_CHAT_ID env vars not set — Telegram disabled.")

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]


# ── Indian FY helper ──────────────────────────────────────────────────────────
def get_fy(dt: datetime) -> str:
    """Returns 'FY2425' style label for a given datetime."""
    y = dt.year
    if dt.month >= 4:
        return f"FY{str(y)[2:]}{str(y+1)[2:]}"
    return f"FY{str(y-1)[2:]}{str(y)[2:]}"


# ── Analyzers ─────────────────────────────────────────────────────────────────
class TradeLogger(bt.Analyzer):
    """
    Records every closed trade with:
      - Correct expense (uses trade.value, not trade.size which = 0 after close)
      - Entry/exit datetime for hold-time calculation
      - FY and calendar year label
    """
    def start(self):
        self.trades   = []
        self._open_dt = {}   # ref → entry datetime

    def notify_trade(self, trade):
        # Track entry time
        if trade.isopen:
            self._open_dt[trade.ref] = self.strategy.data.datetime.datetime(0)
            return
        if not trade.isclosed:
            return

        close_dt       = self.strategy.data.datetime.datetime(0)
        open_dt        = self._open_dt.pop(trade.ref, close_dt)
        hold_bars      = max(1, int((close_dt - open_dt).total_seconds() / 3600))

        gross_pnl      = trade.pnl
        entry_notional = abs(trade.value)
        exit_notional  = abs(entry_notional + gross_pnl)
        expense        = (entry_notional + exit_notional) * CONFIG["EXPENSE_RATE"]
        slippage_cost  = (entry_notional + exit_notional) * CONFIG["SLIPPAGE_PERC"]
        total_cost     = expense + slippage_cost
        net_pnl        = gross_pnl - total_cost

        self.trades.append({
            "dt":          close_dt,
            "open_dt":     open_dt,
            "hold_bars":   hold_bars,
            "fy":          get_fy(close_dt),
            "year":        close_dt.year,
            "month":       close_dt.month,
            "ticker":      self.strategy.data._name,
            "gross_pnl":   gross_pnl,
            "expense":     expense,
            "slippage":    slippage_cost,
            "total_cost":  total_cost,
            "net_pnl":     net_pnl,
            "won":         net_pnl > 0,
            "entry_val":   entry_notional,
        })

    def get_analysis(self):
        return self.trades


class EquityTracker(bt.Analyzer):
    """Bar-by-bar portfolio value — needed for Sharpe / Sortino / Calmar."""
    def start(self):
        self.equity = []

    def next(self):
        self.equity.append(self.strategy.broker.get_value())

    def get_analysis(self):
        return self.equity


# ── Strategy ──────────────────────────────────────────────────────────────────
class AdvancedRSIStrategy(bt.Strategy):
    params = (
        ("rsi_p",   40),
        ("wma_p",   15),
        ("atr_p",   14),
        ("ema_p",  200),   # trend filter — long only above 200 EMA
        ("rr",     2.5),   # reward:risk for full target
        ("atr_m",  2.0),   # ATR multiplier for trailing stop
        ("max_pos",  5),   # max simultaneous open positions
    )

    def __init__(self):
        self.rsi      = bt.indicators.RSI(self.data.close, period=self.p.rsi_p)
        self.wma_rsi  = bt.indicators.WMA(self.rsi,        period=self.p.wma_p)
        self.atr      = bt.indicators.ATR(self.data,       period=self.p.atr_p)
        self.ema200   = bt.indicators.EMA(self.data.close, period=self.p.ema_p)
        self.stop_loss   = None
        self.target      = None
        self.half_booked = False
        self.pending_buy = False   # signal fires on close, entry on next open

    def next(self):
        n_open = sum(1 for d in self.datas if self.getposition(d).size > 0)

        if not self.position:
            # Execute pending signal at next candle's open
            if self.pending_buy:
                self.pending_buy = False
                if n_open < self.p.max_pos:
                    entry = self.data.open[0]
                    risk  = entry - self.data.low[-1]
                    if risk > 0 and entry > 0:
                        qty = int(
                            (self.broker.get_value() * CONFIG["RISK_PERCENT"]) / risk
                        )
                        qty = min(qty, int(self.broker.get_cash() / entry))
                        if qty > 0:
                            self.buy(size=qty)
                            self.stop_loss   = self.data.low[-1]
                            self.target      = entry + risk * self.p.rr
                            self.half_booked = False
                return

            # Signal: RSI × WMA crossover AND price above 200 EMA
            if (self.rsi[0] > self.wma_rsi[0] and
                self.rsi[-1] <= self.wma_rsi[-1] and
                    self.data.close[0] > self.ema200[0]):
                self.pending_buy = True

        else:
            # Partial exit at target (sell half)
            if not self.half_booked and self.data.high[0] >= self.target:
                self.sell(size=int(self.position.size / 2))
                self.half_booked = True
                self.stop_loss   = max(self.stop_loss, self.data.open[0])

            # ATR trailing stop (tightens after partial exit)
            if self.half_booked:
                self.stop_loss = max(
                    self.stop_loss,
                    self.data.close[0] - self.atr[0] * self.p.atr_m,
                )

            # Stop-loss hit → full close
            if self.data.low[0] <= self.stop_loss:
                self.close()


# ── Data helpers ──────────────────────────────────────────────────────────────
def download_data(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """
    Download OHLCV from yfinance.
    NOTE: yfinance caps 1h data at last 730 days regardless of start date.
    For 6-year tests use interval='1d', '1wk', or resample from daily.
    """
    df = yf.download(ticker, start=start, end=end,
                     interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).capitalize() for c in df.columns]
    # Ensure standard column names
    rename = {"Adj close": "Close", "Adj_close": "Close"}
    df.rename(columns=rename, inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def resample_from_daily(daily_df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resample a daily DataFrame to weekly or monthly.
    For sub-day timeframes (2h, 4h) we download hourly and resample.
    """
    tf_map = {"1wk": "W", "1mo": "ME"}
    rule = tf_map.get(tf, tf)
    agg  = {"Open": "first", "High": "max",
             "Low": "min", "Close": "last", "Volume": "sum"}
    out = daily_df.resample(rule).apply(agg).dropna()
    return out[out["Volume"] > 0]


def get_base_interval(tf: str) -> str:
    """Map target timeframe to best yfinance download interval."""
    if tf in ("1h", "2h", "4h"):
        return "1h"      # download 1h, resample up
    return "1d"          # download daily for 1d / 1wk / 1mo


def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample a 1h DataFrame to 2h or 4h."""
    if tf == "1h":
        return df
    tf_map = {"2h": "2h", "4h": "4h", "1d": "D", "1wk": "W"}
    rule = tf_map.get(tf, tf)
    agg  = {"Open": "first", "High": "max",
             "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample(rule).apply(agg).dropna()
    return out[out["Volume"] > 0]


def split_is_oos(df: pd.DataFrame, is_ratio: float = 0.75):
    """Split DataFrame into in-sample / out-of-sample by row count."""
    split_idx = int(len(df) * is_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


# ── Cerebro runner ────────────────────────────────────────────────────────────
def run_cerebro(df: pd.DataFrame, ticker: str, tf: str) -> tuple:
    """
    Run one backtest pass.
    Returns (trades_list, max_drawdown_pct, equity_curve_list).
    """
    min_bars = max(220, 60)   # need 200 bars for EMA200 warmup
    if len(df) < min_bars:
        return [], 0.0, []

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df), name=f"{ticker}_{tf}")
    cerebro.addstrategy(AdvancedRSIStrategy)
    cerebro.broker.setcash(CONFIG["INITIAL_CASH"])
    cerebro.broker.setcommission(commission=0.0)   # handled in TradeLogger
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(TradeLogger,            _name="tl")
    cerebro.addanalyzer(EquityTracker,          _name="eq")

    try:
        results = cerebro.run()
        strat   = results[0]
        dd_raw  = strat.analyzers.dd.get_analysis()
        dd_val  = (dd_raw["max"]["drawdown"]
                   if dd_raw.get("max") and dd_raw["max"].get("drawdown") else 0.0)
        return (strat.analyzers.tl.get_analysis(),
                dd_val,
                strat.analyzers.eq.get_analysis())
    except Exception as e:
        print(f"    [ERROR] {ticker} @ {tf}: {e}")
        return [], 0.0, []


# ── Statistical calculators ───────────────────────────────────────────────────
def annualised_sharpe(equity: list, tf: str) -> float:
    """Risk-free rate = 0 (conservative). Annualised by tf bars/year."""
    bars_per_year = {"1h": 1638, "2h": 819, "4h": 410, "1d": 252,
                     "1wk": 52,  "1mo": 12}.get(tf, 252)
    if len(equity) < 2:
        return 0.0
    arr  = np.array(equity, dtype=float)
    rets = np.diff(arr) / arr[:-1]
    std  = rets.std()
    return float((rets.mean() / std) * np.sqrt(bars_per_year)) if std > 0 else 0.0


def annualised_sortino(equity: list, tf: str) -> float:
    """Sortino uses downside deviation (negative returns only)."""
    bars_per_year = {"1h": 1638, "2h": 819, "4h": 410, "1d": 252,
                     "1wk": 52,  "1mo": 12}.get(tf, 252)
    if len(equity) < 2:
        return 0.0
    arr      = np.array(equity, dtype=float)
    rets     = np.diff(arr) / arr[:-1]
    neg_rets = rets[rets < 0]
    if len(neg_rets) == 0:
        return float("inf")
    downside = neg_rets.std()
    return float((rets.mean() / downside) * np.sqrt(bars_per_year)) if downside > 0 else 0.0


def max_consecutive(trades: list, won: bool) -> int:
    """Longest consecutive win or loss streak."""
    best = cur = 0
    for t in trades:
        if t["won"] == won:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def recovery_factor(net_pnl: float, initial: float, max_dd_pct: float) -> float:
    """Net profit / max absolute drawdown."""
    max_dd_abs = initial * max_dd_pct / 100
    return (net_pnl / max_dd_abs) if max_dd_abs > 0 else 0.0


def build_monthly_table(trades: list) -> str:
    """Build a year × month net P&L table string."""
    grid = defaultdict(lambda: defaultdict(float))
    for t in trades:
        grid[t["year"]][t["month"]] += t["net_pnl"]

    if not grid:
        return "  No data.\n"

    header = "  Year  | " + " | ".join(f"{m:>5}" for m in MONTHS) + " |  Total\n"
    sep    = "  " + "-" * (len(header) - 2) + "\n"
    lines  = header + sep
    for yr in sorted(grid):
        row_vals = [grid[yr].get(m, 0.0) for m in range(1, 13)]
        total    = sum(row_vals)
        cells    = " | ".join(
            f"{'▲' if v >= 0 else '▼'}{abs(v):>4.0f}" for v in row_vals
        )
        lines += f"  {yr}  | {cells} | {total:>+7.0f}\n"
    return lines


def build_yearly_table(trades: list) -> str:
    """Year-by-year P&L, accuracy, trades."""
    yearly = defaultdict(lambda: {"net": 0.0, "wins": 0, "n": 0})
    for t in trades:
        y = t["year"]
        yearly[y]["net"]  += t["net_pnl"]
        yearly[y]["wins"] += int(t["won"])
        yearly[y]["n"]    += 1

    lines = ""
    for yr in sorted(yearly):
        d   = yearly[yr]
        acc = d["wins"] / d["n"] * 100 if d["n"] > 0 else 0
        sign = "+" if d["net"] >= 0 else ""
        lines += (
            f"  {yr} → Net: {sign}₹{d['net']:>10,.0f}  "
            f"Acc: {acc:.1f}%  Trades: {d['n']}\n"
        )
    return lines


def build_fy_tax(trades: list) -> tuple:
    """Returns (fy_lines_str, total_stcg_tax)."""
    fy = defaultdict(lambda: {"wins": 0, "losses": 0,
                               "gross_p": 0.0, "gross_l": 0.0,
                               "expenses": 0.0})
    for t in trades:
        k = t["fy"]
        fy[k]["expenses"] += t["total_cost"]
        if t["won"]:
            fy[k]["wins"]    += 1
            fy[k]["gross_p"] += t["gross_pnl"]
        else:
            fy[k]["losses"]  += 1
            fy[k]["gross_l"] += abs(t["gross_pnl"])

    total_tax = 0.0
    lines     = ""
    for k in sorted(fy):
        d      = fy[k]
        fy_net = d["gross_p"] - d["gross_l"] - d["expenses"]
        stcg   = max(0.0, fy_net * 0.15)
        total_tax += stcg
        n_trades = d["wins"] + d["losses"]
        acc  = d["wins"] / n_trades * 100 if n_trades > 0 else 0
        lines += (
            f"  *{k}*\n"
            f"    Wins  : +₹{d['gross_p']:>12,.0f}  Losses: −₹{d['gross_l']:>12,.0f}\n"
            f"    Expense: −₹{d['expenses']:>10,.0f}  Net: ₹{fy_net:>10,.0f}\n"
            f"    STCG(15%): −₹{stcg:>8,.0f}  Acc: {acc:.1f}%\n\n"
        )
    return lines, total_tax


# ── Full stats for one trade set ──────────────────────────────────────────────
def calc_stats(trades: list, equity: list, max_dd: float,
               label: str, tf: str, total_calendar_days: int) -> dict:
    if not trades:
        return {"label": label, "tf": tf, "empty": True}

    n        = len(trades)
    wins     = [t for t in trades if t["won"]]
    losses   = [t for t in trades if not t["won"]]
    gross_p  = sum(t["gross_pnl"] for t in wins)
    gross_l  = sum(abs(t["gross_pnl"]) for t in losses)
    expenses = sum(t["expense"] for t in trades)
    slip     = sum(t["slippage"] for t in trades)
    net_pnl  = sum(t["net_pnl"] for t in trades)
    roi      = net_pnl / CONFIG["INITIAL_CASH"] * 100

    avg_win  = gross_p / len(wins)   if wins   else 0.0
    avg_loss = gross_l / len(losses) if losses else 0.0
    rr       = avg_win / avg_loss    if avg_loss > 0 else 0.0
    pf       = gross_p / gross_l     if gross_l > 0 else float("inf")
    wr       = len(wins) / n * 100
    exp      = net_pnl / n

    avg_hold = sum(t["hold_bars"] for t in trades) / n
    w_streak = max_consecutive(trades, True)
    l_streak = max_consecutive(trades, False)

    # NSE trading days in the window
    trading_days = max(1, total_calendar_days * 252 // 365)
    tpd          = n / trading_days

    sharpe  = annualised_sharpe(equity, tf)
    sortino = annualised_sortino(equity, tf)
    calmar  = roi / max_dd if max_dd > 0 else 0.0
    rec_fac = recovery_factor(net_pnl, CONFIG["INITIAL_CASH"], max_dd)

    fy_lines, total_tax = build_fy_tax(trades)
    net_after_tax = net_pnl - total_tax

    return {
        "label": label, "tf": tf, "empty": False,
        "n": n, "wins": len(wins), "losses": len(losses),
        "gross_p": gross_p, "gross_l": gross_l,
        "expenses": expenses, "slippage": slip,
        "net_pnl": net_pnl, "roi": roi,
        "avg_win": avg_win, "avg_loss": avg_loss, "rr": rr,
        "pf": pf, "wr": wr, "exp": exp,
        "max_dd": max_dd,
        "sharpe": sharpe, "sortino": sortino,
        "calmar": calmar, "rec_fac": rec_fac,
        "avg_hold": avg_hold,
        "w_streak": w_streak, "l_streak": l_streak,
        "tpd": tpd,
        "total_tax": total_tax, "net_after_tax": net_after_tax,
        "fy_lines": fy_lines,
        "trades_list": trades,   # kept for monthly/yearly table generation
        "equity": equity,
    }


# ── Report formatter ──────────────────────────────────────────────────────────
def flag(val, good_above, warn_above, hi_good=True):
    """Return emoji flag based on threshold direction."""
    if hi_good:
        return "✅" if val >= good_above else ("🟡" if val >= warn_above else "🔴")
    else:  # lower is better
        return "✅" if val <= warn_above else ("🟡" if val <= good_above else "🔴")


def fmt_stats_block(s: dict) -> str:
    if s.get("empty"):
        return "  No trades in this window.\n\n"

    dd_comment = (
        "🔴 > 30% — too large for live trading" if s["max_dd"] > 30 else
        "🟡 20–30% — use position sizing carefully" if s["max_dd"] > 20 else
        "🟢 < 20% — acceptable"
    )
    freq_note = (
        f"⚠️ {s['tpd']:.1f}/day — overtrading" if s["tpd"] > 5
        else f"{s['tpd']:.2f}/day ✅"
    )

    return (
        f"*P/L Summary*\n"
        f"  Gross wins       : +₹{s['gross_p']:>13,.2f}\n"
        f"  Gross losses     : −₹{s['gross_l']:>13,.2f}\n"
        f"  Gross P/L        :  ₹{s['gross_p']-s['gross_l']:>13,.2f}\n"
        f"  Expenses (0.4%)  : −₹{s['expenses']:>13,.2f}\n"
        f"  Slippage (0.1%)  : −₹{s['slippage']:>13,.2f}\n"
        f"  *Net realised P/L*:  *₹{s['net_pnl']:>11,.2f}*\n"
        f"  STCG tax (15%)   : −₹{s['total_tax']:>13,.2f}\n"
        f"  *Net after tax*  :  *₹{s['net_after_tax']:>11,.2f}*\n"
        f"  ROI on capital   :  {s['roi']:>+.2f}%\n\n"

        f"*Trade Statistics*\n"
        f"  Total trades  : {s['n']}  (Wins: {s['wins']}  Losses: {s['losses']})\n"
        f"  Accuracy      : {s['wr']:.1f}%  {flag(s['wr'],55,45)}\n"
        f"  Profit Factor : {s['pf']:.2f}  {flag(s['pf'],1.5,1.0)}\n"
        f"  Expectancy    : ₹{s['exp']:.0f}/trade  {flag(s['exp'],0,0)}\n"
        f"  Avg Win       : ₹{s['avg_win']:.0f}\n"
        f"  Avg Loss      : ₹{s['avg_loss']:.0f}\n"
        f"  Realised R:R  : {s['rr']:.2f}x  {flag(s['rr'],1.5,1.0)}\n"
        f"  Avg Hold      : {s['avg_hold']:.1f} bars\n"
        f"  Win streak    : {s['w_streak']}  |  Loss streak: {s['l_streak']}\n"
        f"  Trade freq    : {freq_note}\n\n"

        f"*Risk Metrics*\n"
        f"  Max Drawdown    : {s['max_dd']:.2f}%  {dd_comment}\n"
        f"  Sharpe Ratio    : {s['sharpe']:.2f}  {flag(s['sharpe'],1.5,1.0)}\n"
        f"  Sortino Ratio   : {s['sortino']:.2f}  {flag(s['sortino'],2.0,1.0)}\n"
        f"  Calmar Ratio    : {s['calmar']:.2f}  {flag(s['calmar'],1.0,0.5)}\n"
        f"  Recovery Factor : {s['rec_fac']:.2f}  {flag(s['rec_fac'],2.0,1.0)}\n\n"
    )


def fmt_tf_report(tf: str, is_s: dict, oos_s: dict,
                  best_sym: str, best_pnl: float,
                  all_trades: list, start: str, end: str) -> str:
    """Build full per-timeframe Telegram report string."""

    # Walk-forward verdict
    wf = ""
    if not is_s.get("empty") and not oos_s.get("empty"):
        drop = is_s["wr"] - oos_s["wr"]
        wf   = (
            f"\n*── WALK-FORWARD VERDICT ──*\n"
            f"  IS  → Acc:{is_s['wr']:.1f}%  Net:₹{is_s['net_pnl']:,.0f}\n"
            f"  OOS → Acc:{oos_s['wr']:.1f}%  Net:₹{oos_s['net_pnl']:,.0f}\n"
        )
        if oos_s["net_pnl"] < 0:
            wf += "  🔴 OOS net negative — edge may not be real.\n"
        elif drop > 5:
            wf += f"  ⚠️ {drop:.1f}% accuracy drop OOS — possible curve-fit.\n"
        elif oos_s["wr"] >= is_s["wr"] - 3:
            wf += "  ✅ OOS robust — edge holds out-of-sample.\n"
        else:
            wf += f"  🟡 Minor degradation ({drop:.1f}%). Monitor live.\n"

    # Yearly table
    yearly_tbl = build_yearly_table(all_trades)

    # FY tax (combined)
    fy_lines, _ = build_fy_tax(all_trades)

    return (
        f"📊 *TIMEFRAME REPORT — {tf.upper()}*\n"
        f"Period: {start} → {end}\n"
        f"Capital: ₹{CONFIG['INITIAL_CASH']:,}  |  "
        f"Costs: 0.4% exp + 0.1% slip/side\n"
        f"Strategy: AdvancedRSI + EMA200 filter\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        f"📈 *IN-SAMPLE ({is_s['label']})*\n"
        f"{fmt_stats_block(is_s)}"

        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        f"🧪 *OUT-OF-SAMPLE ({oos_s['label']})*\n"
        f"{fmt_stats_block(oos_s)}"

        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        f"{wf}\n"

        f"*── YEAR-BY-YEAR P/L ──*\n"
        f"{yearly_tbl}\n"

        f"*── FY TAX BUCKETS (combined) ──*\n"
        f"{fy_lines}"

        f"*── STAR PERFORMER ──*\n"
        f"  {best_sym}  (+₹{best_pnl:,.0f})\n\n"

        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"_Losses set off vs gains same FY (Indian IT Act)._\n"
        f"_STCG @ 15% flat. Loss carry-forward: 8 yrs._\n"
        f"_Consult CA before filing. Auto-generated._"
    )


def fmt_master_summary(tf_results: dict, start: str, end: str) -> str:
    """Cross-timeframe comparison summary."""
    lines = (
        f"🏛 *MASTER SUMMARY — ALL TIMEFRAMES*\n"
        f"Period: {start} → {end}\n"
        f"Capital: ₹{CONFIG['INITIAL_CASH']:,}  |  Strategy: AdvancedRSI + EMA200\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{'TF':<5} {'Net P/L':>10} {'ROI%':>7} {'Acc%':>6} "
        f"{'PF':>5} {'Sharpe':>7} {'Sortino':>8} "
        f"{'MaxDD%':>7} {'Calmar':>7} {'Trades':>7}\n"
        f"{'─'*5} {'─'*10} {'─'*7} {'─'*6} "
        f"{'─'*5} {'─'*7} {'─'*8} "
        f"{'─'*7} {'─'*7} {'─'*7}\n"
    )

    best_tf     = None
    best_sharpe = -999

    for tf, res in tf_results.items():
        s = res["combined"]
        if s.get("empty"):
            lines += f"{tf:<5} {'no trades':>10}\n"
            continue
        lines += (
            f"{tf:<5} "
            f"₹{s['net_pnl']:>9,.0f} "
            f"{s['roi']:>+6.1f}% "
            f"{s['wr']:>5.1f}% "
            f"{s['pf']:>5.2f} "
            f"{s['sharpe']:>7.2f} "
            f"{s['sortino']:>8.2f} "
            f"{s['max_dd']:>6.2f}% "
            f"{s['calmar']:>7.2f} "
            f"{s['n']:>7}\n"
        )
        if s["sharpe"] > best_sharpe:
            best_sharpe = s["sharpe"]
            best_tf     = tf

    lines += (
        f"\n{'─'*75}\n"
        f"🏆 Best risk-adjusted timeframe: *{best_tf}* "
        f"(Sharpe: {best_sharpe:.2f})\n\n"
        f"*Metric guide*\n"
        f"  Sharpe > 1.5 ✅  |  Sortino > 2.0 ✅  |  Calmar > 1.0 ✅\n"
        f"  Max DD < 20% ✅  |  PF > 1.5 ✅       |  Acc > 55% ✅\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"_STCG @ 15%. Losses set off same FY. Consult CA._\n"
        f"_Auto-generated. Not financial advice._"
    )
    return lines


# ── Telegram helpers ──────────────────────────────────────────────────────────
def send_msg(text: str, chunk_size: int = 4000):
    """Send message, auto-splitting if > Telegram 4096 char limit."""
    if not CONFIG["TOKEN"] or not CONFIG["CHAT_ID"]:
        return
    url = f"https://api.telegram.org/bot{CONFIG['TOKEN']}/sendMessage"
    # Split at newline boundaries to avoid breaking Markdown
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > chunk_size:
            chunks.append(buf)
            buf = ""
        buf += line
    if buf:
        chunks.append(buf)

    for chunk in chunks:
        try:
            requests.post(
                url,
                json={"chat_id": CONFIG["CHAT_ID"],
                      "text": chunk, "parse_mode": "Markdown"},
                timeout=10,
            )
            time.sleep(0.5)   # avoid Telegram rate limit
        except Exception as e:
            print(f"[Telegram] {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    START = CONFIG["START_DATE"]
    END   = CONFIG["END_DATE"]

    # Calendar days in backtest window (for trade-frequency calculation)
    cal_days = (datetime.strptime(END, "%Y-%m-%d") -
                datetime.strptime(START, "%Y-%m-%d")).days

    with open(CONFIG["WATCHLIST_FILE"], "r") as f:
        symbols = [s.strip().upper() for s in f.read().splitlines() if s.strip()]

    # Structure: tf_data[tf] = {
    #   "is_trades":[], "oos_trades":[], "all_trades":[],
    #   "is_equity":[], "oos_equity":[],
    #   "is_max_dd": float, "oos_max_dd": float, "all_max_dd": float,
    #   "best": {"name":"", "profit": -999}
    # }
    tf_data = {}
    for tf in CONFIG["TEST_TIMEFRAMES"]:
        tf_data[tf] = {
            "is_trades": [], "oos_trades": [], "all_trades": [],
            "is_equity": [], "oos_equity": [], "all_equity": [],
            "is_max_dd": 0.0, "oos_max_dd": 0.0, "all_max_dd": 0.0,
            "best": {"name": "", "profit": -999_999},
        }

    # ── Download & run ────────────────────────────────────────────────────────
    for symbol in symbols:
        ticker = symbol if "." in symbol else f"{symbol}.NS"
        print(f"\n[{ticker}]")

        # Determine base download interval
        # For 2h / 4h we need hourly data; yfinance limits 1h to last 730 days.
        # So for the 6-year window, 2h/4h will only have data from 2023 onward.
        # 1d is available for the full range.
        base_interval = get_base_interval(CONFIG["TEST_TIMEFRAMES"][0])

        # Download hourly if any sub-day TF requested
        needs_hourly = any(tf in ("1h", "2h", "4h")
                           for tf in CONFIG["TEST_TIMEFRAMES"])
        needs_daily  = any(tf in ("1d", "1wk")
                           for tf in CONFIG["TEST_TIMEFRAMES"])

        hourly_df = pd.DataFrame()
        daily_df  = pd.DataFrame()

        if needs_hourly:
            hourly_df = download_data(ticker, START, END, "1h")
            if not hourly_df.empty:
                print(f"  Hourly bars: {len(hourly_df)}")

        if needs_daily:
            daily_df = download_data(ticker, START, END, "1d")
            if not daily_df.empty:
                print(f"  Daily bars : {len(daily_df)}")

        for tf in CONFIG["TEST_TIMEFRAMES"]:
            print(f"  [{tf}] ...", end="", flush=True)
            d = tf_data[tf]

            # Choose and resample base data
            if tf in ("1h", "2h", "4h"):
                if hourly_df.empty:
                    print(" skip (no hourly data)")
                    continue
                df = resample_ohlcv(hourly_df.copy(), tf)
            else:
                if daily_df.empty:
                    print(" skip (no daily data)")
                    continue
                df = resample_ohlcv(daily_df.copy(), tf) if tf != "1d" else daily_df.copy()

            if len(df) < 220:
                print(f" skip (only {len(df)} bars — need 220 for EMA200 warmup)")
                continue

            # Split IS / OOS
            is_df, oos_df = split_is_oos(df, CONFIG["IS_RATIO"])

            # Run in-sample
            t_is,  dd_is,  eq_is  = run_cerebro(is_df,  ticker, tf)
            # Run out-of-sample
            t_oos, dd_oos, eq_oos = run_cerebro(oos_df, ticker, tf)
            # Run full window (for combined stats)
            t_all, dd_all, eq_all = run_cerebro(df,     ticker, tf)

            d["is_trades"].extend(t_is);   d["is_equity"].extend(eq_is)
            d["oos_trades"].extend(t_oos); d["oos_equity"].extend(eq_oos)
            d["all_trades"].extend(t_all); d["all_equity"].extend(eq_all)
            d["is_max_dd"]  = max(d["is_max_dd"],  dd_is)
            d["oos_max_dd"] = max(d["oos_max_dd"], dd_oos)
            d["all_max_dd"] = max(d["all_max_dd"], dd_all)

            sym_net = sum(t["net_pnl"] for t in t_all) if t_all else 0
            if sym_net > d["best"]["profit"]:
                d["best"] = {"name": f"{ticker} ({tf})", "profit": sym_net}

            print(f" IS:{len(t_is)} OOS:{len(t_oos)} ALL:{len(t_all)}")

        time.sleep(CONFIG["DELAY_SECONDS"])

    # ── Generate reports per timeframe ────────────────────────────────────────
    tf_results = {}   # for master summary

    for tf in CONFIG["TEST_TIMEFRAMES"]:
        d = tf_data[tf]
        print(f"\n[Report] {tf} ...")

        is_s  = calc_stats(d["is_trades"],  d["is_equity"],
                           d["is_max_dd"],  f"IS 75%",  tf, int(cal_days * 0.75))
        oos_s = calc_stats(d["oos_trades"], d["oos_equity"],
                           d["oos_max_dd"], f"OOS 25%", tf, int(cal_days * 0.25))
        all_s = calc_stats(d["all_trades"], d["all_equity"],
                           d["all_max_dd"], f"Full {START[:7]}→{END[:7]}", tf, cal_days)

        tf_results[tf] = {"is": is_s, "oos": oos_s, "combined": all_s}

        if not d["all_trades"]:
            print(f"  No trades for {tf} — skipping report.")
            continue

        report = fmt_tf_report(
            tf, is_s, oos_s,
            d["best"]["name"], d["best"]["profit"],
            d["all_trades"], START, END,
        )

        print(f"\n{'='*65}")
        print(report)
        print(f"{'='*65}")
        send_msg(report)
        time.sleep(2)   # Telegram rate-limit gap between messages

    # ── Master cross-timeframe summary ────────────────────────────────────────
    master = fmt_master_summary(tf_results, START, END)
    print(f"\n{'='*65}")
    print(master)
    print(f"{'='*65}")
    send_msg(master)

    print("\n[Done] All reports sent.")
