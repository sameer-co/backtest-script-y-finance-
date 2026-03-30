"""
AdvancedRSI Backtesting Engine — Professional Edition v2
=========================================================
Fixes vs v1:
  - Expense calculation bug fixed (trade.size = 0 after close)
    Now uses trade.value (entry notional) + pnl to derive exit notional
  - Added Calmar ratio, Sharpe approximation, trade frequency check
  - Added max open positions guard (default 5)
  - Added next-candle entry (signal on close, execute on next open)
  - Added 0.1% slippage model on top of 0.4% expense
  - Added 200-EMA trend filter (long only above EMA)
  - Added walk-forward split (first 18m in-sample, last 6m OOS test)
  - Telegram token raises warning if env var missing — no hardcoded fallback
"""

import backtrader as bt
import yfinance as yf
import pandas as pd
import os, requests, time, collections, collections.abc
import numpy as np
from datetime import datetime

# ── Compatibility ────────────────────────────────────────────────────────────
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    # Set TG_TOKEN and TG_CHAT_ID as environment variables
    "TOKEN":            os.getenv("TG_TOKEN"),
    "CHAT_ID":          os.getenv("TG_CHAT_ID"),

    "INITIAL_CASH":     50_000,
    "RISK_PERCENT":     0.02,      # 2% of portfolio equity per trade
    "EXPENSE_RATE":     0.004,     # 0.4% per side — STT + brokerage
    "SLIPPAGE_PERC":    0.001,     # 0.1% per side — realistic slippage
    "MAX_POSITIONS":    5,         # max simultaneous open positions
    "BACKTEST_PERIOD":  "2y",
    "WATCHLIST_FILE":   "watchlist.txt",
    "TEST_TIMEFRAMES":  ["2h"],
    "DELAY_SECONDS":    5,
    "IS_MONTHS":        18,        # in-sample window (months)
    "OOS_MONTHS":       6,         # out-of-sample window (months)
}

if not CONFIG["TOKEN"] or not CONFIG["CHAT_ID"]:
    print("[WARNING] TG_TOKEN or TG_CHAT_ID not set. Telegram reporting disabled.")


# ── Indian FY helper ─────────────────────────────────────────────────────────
def get_fy(dt: datetime) -> str:
    if dt.month >= 4:
        return f"FY{str(dt.year)[2:]}{str(dt.year+1)[2:]}"
    return f"FY{str(dt.year-1)[2:]}{str(dt.year)[2:]}"


# ── Trade Logger Analyzer ────────────────────────────────────────────────────
class TradeLogger(bt.Analyzer):
    """
    Captures every closed trade with correct expense calculation.

    Bug fix: trade.size == 0 after a trade closes in backtrader.
    Solution: use abs(trade.value) as entry notional (preserved by bt),
    then derive exit notional from entry_notional + gross_pnl.
    """

    def start(self):
        self.trades = []

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        dt             = self.strategy.data.datetime.datetime(0)
        gross_pnl      = trade.pnl
        entry_notional = abs(trade.value)
        exit_notional  = abs(entry_notional + gross_pnl)
        expense        = (entry_notional + exit_notional) * CONFIG["EXPENSE_RATE"]
        slippage_cost  = (entry_notional + exit_notional) * CONFIG["SLIPPAGE_PERC"]
        total_cost     = expense + slippage_cost
        net_pnl        = gross_pnl - total_cost

        self.trades.append({
            "dt":         dt,
            "fy":         get_fy(dt),
            "ticker":     self.strategy.data._name,
            "gross_pnl":  gross_pnl,
            "expense":    expense,
            "slippage":   slippage_cost,
            "total_cost": total_cost,
            "net_pnl":    net_pnl,
            "won":        net_pnl > 0,
            "entry_val":  entry_notional,
        })

    def get_analysis(self):
        return self.trades


# ── Equity Curve Tracker ─────────────────────────────────────────────────────
class EquityTracker(bt.Analyzer):
    def start(self):
        self.equity = []

    def next(self):
        self.equity.append(self.strategy.broker.get_value())

    def get_analysis(self):
        return self.equity


# ── Strategy ─────────────────────────────────────────────────────────────────
class AdvancedRSIStrategy(bt.Strategy):
    params = (
        ("rsi_p",   40),
        ("wma_p",   15),
        ("atr_p",   14),
        ("ema_p",  200),   # trend filter period
        ("rr",     2.5),   # reward:risk ratio for target
        ("atr_m",  2.0),   # ATR multiplier for trailing stop
        ("max_pos",  5),   # max simultaneous positions
    )

    def __init__(self):
        self.rsi     = bt.indicators.RSI(self.data.close, period=self.p.rsi_p)
        self.wma_rsi = bt.indicators.WMA(self.rsi,        period=self.p.wma_p)
        self.atr     = bt.indicators.ATR(self.data,       period=self.p.atr_p)
        self.ema200  = bt.indicators.EMA(self.data.close, period=self.p.ema_p)

        self.stop_loss   = None
        self.target      = None
        self.half_booked = False
        self.pending_buy = False   # next-candle entry

    def next(self):
        n_open = sum(1 for d in self.datas if self.getposition(d).size > 0)

        if not self.position:
            # Execute pending signal on next candle's open
            if self.pending_buy:
                self.pending_buy = False
                if n_open < self.p.max_pos:
                    entry = self.data.open[0]
                    risk  = entry - self.data.low[-1]
                    if risk > 0 and entry > 0:
                        qty = int(
                            (self.broker.get_value() * CONFIG["RISK_PERCENT"]) / risk
                        )
                        final_qty = min(qty, int(self.broker.get_cash() / entry))
                        if final_qty > 0:
                            self.buy(size=final_qty)
                            self.stop_loss   = self.data.low[-1]
                            self.target      = entry + (risk * self.p.rr)
                            self.half_booked = False
                return

            # Signal: RSI crosses WMA AND price is above 200 EMA
            rsi_cross = (
                self.rsi[0] > self.wma_rsi[0] and
                self.rsi[-1] <= self.wma_rsi[-1]
            )
            above_ema = self.data.close[0] > self.ema200[0]

            if rsi_cross and above_ema:
                self.pending_buy = True

        elif self.position:
            # Partial exit at target
            if not self.half_booked and self.data.high[0] >= self.target:
                self.sell(size=int(self.position.size / 2))
                self.half_booked = True
                self.stop_loss   = max(self.stop_loss, self.data.open[0])

            # ATR trailing stop
            if self.half_booked:
                self.stop_loss = max(
                    self.stop_loss,
                    self.data.close[0] - (self.atr[0] * self.p.atr_m),
                )

            # Stop hit
            if self.data.low[0] <= self.stop_loss:
                self.close()


# ── Helpers ───────────────────────────────────────────────────────────────────
def send_msg(text: str):
    if not CONFIG["TOKEN"] or not CONFIG["CHAT_ID"]:
        return
    url = f"https://api.telegram.org/bot{CONFIG['TOKEN']}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": CONFIG["CHAT_ID"], "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"[Telegram] {e}")


def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "1h":
        return df
    tf_map = {"1d": "D", "4h": "4h", "2h": "2h", "1w": "W"}
    rule = tf_map.get(timeframe, timeframe)
    agg  = {"Open": "first", "High": "max",
             "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample(rule).apply(agg).dropna()
    return out[out["Volume"] > 0]


def split_oos(df: pd.DataFrame, is_months: int):
    if df.empty:
        return df, pd.DataFrame()
    cutoff = df.index[0] + pd.DateOffset(months=is_months)
    return df[df.index < cutoff], df[df.index >= cutoff]


def sharpe_ratio(equity: list, periods_per_year: int = 1095) -> float:
    if len(equity) < 2:
        return 0.0
    arr  = np.array(equity)
    rets = np.diff(arr) / arr[:-1]
    std  = rets.std()
    return float((rets.mean() / std) * np.sqrt(periods_per_year)) if std > 0 else 0.0


def run_cerebro(df: pd.DataFrame, ticker: str, tf: str):
    """Run one cerebro pass. Returns (trades, max_dd, equity)."""
    if len(df) < 60:
        return [], 0.0, []
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df), name=f"{ticker}_{tf}")
    cerebro.addstrategy(AdvancedRSIStrategy)
    cerebro.broker.setcash(CONFIG["INITIAL_CASH"])
    cerebro.broker.setcommission(commission=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(TradeLogger,            _name="tl")
    cerebro.addanalyzer(EquityTracker,          _name="eq")
    try:
        res   = cerebro.run()
        strat = res[0]
        dd_raw = strat.analyzers.dd.get_analysis()
        dd_val = (dd_raw["max"]["drawdown"]
                  if dd_raw.get("max") and dd_raw["max"].get("drawdown") else 0.0)
        return (strat.analyzers.tl.get_analysis(),
                dd_val,
                strat.analyzers.eq.get_analysis())
    except Exception as e:
        print(f"    [ERROR] {ticker} @ {tf}: {e}")
        return [], 0.0, []


def build_fy_summary(trades: list) -> dict:
    fy = {}
    for t in trades:
        k = t["fy"]
        if k not in fy:
            fy[k] = {"wins": 0, "losses": 0,
                     "gross_p": 0.0, "gross_l": 0.0,
                     "expenses": 0.0, "net_pnl": 0.0}
        b = fy[k]
        b["expenses"] += t["total_cost"]
        if t["won"]:
            b["wins"]    += 1
            b["gross_p"] += t["gross_pnl"]
        else:
            b["losses"]  += 1
            b["gross_l"] += abs(t["gross_pnl"])
        b["net_pnl"] += t["net_pnl"]
    return fy


def calc_stats(trades: list, equity: list, max_dd: float, label: str) -> dict:
    if not trades:
        return {"label": label, "empty": True}

    n       = len(trades)
    wins    = [t for t in trades if t["won"]]
    losses  = [t for t in trades if not t["won"]]
    gross_p = sum(t["gross_pnl"] for t in wins)
    gross_l = sum(abs(t["gross_pnl"]) for t in losses)
    expenses= sum(t["expense"] for t in trades)
    slip    = sum(t["slippage"] for t in trades)
    net_pnl = sum(t["net_pnl"] for t in trades)
    roi     = net_pnl / CONFIG["INITIAL_CASH"] * 100

    avg_win  = gross_p / len(wins)   if wins   else 0.0
    avg_loss = gross_l / len(losses) if losses else 0.0
    rr       = avg_win / avg_loss    if avg_loss > 0 else 0.0
    pf       = gross_p / gross_l     if gross_l > 0 else float("inf")
    wr       = len(wins) / n * 100
    exp      = net_pnl / n
    sh       = sharpe_ratio(equity)
    cal      = (roi / max_dd)        if max_dd > 0 else 0.0

    # Trade frequency (NSE ~252 days/yr → 504 in 2y; IS=378, OOS=126 approx)
    trading_days = 378 if "18" in label else 126
    tpd = n / trading_days

    # FY tax buckets
    fy_data   = build_fy_summary(trades)
    total_tax = 0.0
    fy_lines  = ""
    for fy_k, d in sorted(fy_data.items()):
        fy_net = d["gross_p"] - d["gross_l"] - d["expenses"]
        stcg   = max(0.0, fy_net * 0.15)
        total_tax += stcg
        fy_acc = (d["wins"] / (d["wins"] + d["losses"]) * 100
                  if (d["wins"] + d["losses"]) > 0 else 0)
        fy_lines += (
            f"  *{fy_k}*\n"
            f"    Wins: +₹{d['gross_p']:>12,.0f}  |  Losses: −₹{d['gross_l']:>12,.0f}\n"
            f"    Expenses: −₹{d['expenses']:>8,.0f}  |  Net: ₹{fy_net:>10,.0f}\n"
            f"    STCG (15%): −₹{stcg:>8,.0f}  |  Acc: {fy_acc:.1f}%\n\n"
        )

    return {
        "label": label, "empty": False,
        "n": n, "wins": len(wins), "losses": len(losses),
        "gross_p": gross_p, "gross_l": gross_l,
        "expenses": expenses, "slippage": slip,
        "net_pnl": net_pnl, "roi": roi,
        "avg_win": avg_win, "avg_loss": avg_loss, "rr": rr,
        "pf": pf, "wr": wr, "exp": exp,
        "max_dd": max_dd, "sharpe": sh, "calmar": cal,
        "total_tax": total_tax, "net_after_tax": net_pnl - total_tax,
        "fy_lines": fy_lines, "tpd": tpd,
    }


def fmt_section(s: dict) -> str:
    if s.get("empty"):
        return "  No trades in this window.\n\n"

    dd_flag = (
        "🔴 > 30% — not tradeable live"          if s["max_dd"] > 30 else
        "🟡 20–30% — manageable with discipline" if s["max_dd"] > 20 else
        "🟢 < 20% — acceptable"
    )
    cal_flag = "⚠️ < 1.0 — drawdown outpacing returns" if s["calmar"] < 1 else "✅"
    sh_flag  = "⚠️ < 1.0 — poor risk-adjusted return"  if s["sharpe"]  < 1 else "✅"
    freq_warn= (f"⚠️ {s['tpd']:.1f}/day — overtrading, tighten filters"
                if s["tpd"] > 5 else f"{s['tpd']:.1f}/day")

    return (
        f"*P/L Breakdown*\n"
        f"  Gross wins     : +₹{s['gross_p']:>12,.2f}\n"
        f"  Gross losses   : −₹{s['gross_l']:>12,.2f}\n"
        f"  Gross P/L      :  ₹{s['gross_p']-s['gross_l']:>12,.2f}\n"
        f"  Expenses (0.4%): −₹{s['expenses']:>12,.2f}\n"
        f"  Slippage (0.1%): −₹{s['slippage']:>12,.2f}\n"
        f"  *Net P/L*      :  *₹{s['net_pnl']:>10,.2f}*\n"
        f"  STCG tax (15%) : −₹{s['total_tax']:>12,.2f}\n"
        f"  *Net after tax*:  *₹{s['net_after_tax']:>10,.2f}*\n"
        f"  ROI on capital :  {s['roi']:.2f}%\n\n"
        f"*Strategy Metrics*\n"
        f"  Trades     : {s['n']}  (W:{s['wins']} L:{s['losses']})\n"
        f"  Accuracy   : {s['wr']:.1f}%\n"
        f"  Profit Factor : {s['pf']:.2f}\n"
        f"  Expectancy : ₹{s['exp']:.0f}/trade\n"
        f"  Avg Win    : ₹{s['avg_win']:.0f}  |  Avg Loss: ₹{s['avg_loss']:.0f}\n"
        f"  Realised R:R : {s['rr']:.2f}x\n\n"
        f"*Risk Metrics*\n"
        f"  Max Drawdown : {s['max_dd']:.2f}%  {dd_flag}\n"
        f"  Sharpe Ratio : {s['sharpe']:.2f}  {sh_flag}\n"
        f"  Calmar Ratio : {s['calmar']:.2f}  {cal_flag}\n"
        f"  Trade freq   : {freq_warn}\n\n"
        f"*FY Tax Buckets*\n"
        f"{s['fy_lines']}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    with open(CONFIG["WATCHLIST_FILE"], "r") as f:
        symbols = [s.strip().upper() for s in f.read().splitlines() if s.strip()]

    is_trades,  oos_trades  = [], []
    is_equity,  oos_equity  = [], []
    is_max_dd,  oos_max_dd  = 0.0, 0.0
    tf_stats = {tf: {"profit": 0.0, "trades": 0, "wins": 0}
                for tf in CONFIG["TEST_TIMEFRAMES"]}
    best = {"name": "", "profit": -999_999}

    for symbol in symbols:
        ticker = symbol if "." in symbol else f"{symbol}.NS"
        raw_df = yf.download(ticker, period=CONFIG["BACKTEST_PERIOD"],
                             interval="1h", progress=False)
        if raw_df.empty:
            print(f"[SKIP] No data — {ticker}")
            continue
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        raw_df.columns = [str(c) for c in raw_df.columns]
        if len(raw_df) < 80:
            print(f"[SKIP] Too few bars — {ticker}")
            continue

        for tf in CONFIG["TEST_TIMEFRAMES"]:
            print(f"  {ticker} @ {tf} ...", end="", flush=True)
            df = resample_data(raw_df.copy(), tf)
            is_df, oos_df = split_oos(df, CONFIG["IS_MONTHS"])

            t_is,  dd_is,  eq_is  = run_cerebro(is_df,  ticker, tf)
            t_oos, dd_oos, eq_oos = run_cerebro(oos_df, ticker, tf)

            is_trades.extend(t_is);   is_equity.extend(eq_is)
            oos_trades.extend(t_oos); oos_equity.extend(eq_oos)
            is_max_dd  = max(is_max_dd,  dd_is)
            oos_max_dd = max(oos_max_dd, dd_oos)

            all_sym = t_is + t_oos
            if all_sym:
                sym_net = sum(t["net_pnl"] for t in all_sym)
                tf_stats[tf]["profit"] += sym_net
                tf_stats[tf]["trades"] += len(all_sym)
                tf_stats[tf]["wins"]   += sum(1 for t in all_sym if t["won"])
                if sym_net > best["profit"]:
                    best = {"name": f"{ticker} ({tf})", "profit": sym_net}

            print(f" IS={len(t_is)} | OOS={len(t_oos)}")

        time.sleep(CONFIG["DELAY_SECONDS"])

    # ── Stats ──
    is_s  = calc_stats(is_trades,  is_equity,  is_max_dd,  "In-Sample 18m")
    oos_s = calc_stats(oos_trades, oos_equity, oos_max_dd, "Out-of-Sample 6m")

    # ── Timeframe lines ──
    tf_lines = ""
    for tf, d in tf_stats.items():
        acc = (d["wins"] / d["trades"] * 100) if d["trades"] > 0 else 0
        tf_lines += f"  *{tf}*: ₹{d['profit']:,.0f}  Acc:{acc:.1f}%  Trades:{d['trades']}\n"

    # ── Walk-forward verdict ──
    wf_verdict = ""
    if not is_s.get("empty") and not oos_s.get("empty"):
        drop = is_s["wr"] - oos_s["wr"]
        wf_verdict = (
            f"\n*── WALK-FORWARD VERDICT ──*\n"
            f"  IS  → Acc:{is_s['wr']:.1f}%  Net:₹{is_s['net_pnl']:,.0f}\n"
            f"  OOS → Acc:{oos_s['wr']:.1f}%  Net:₹{oos_s['net_pnl']:,.0f}\n"
        )
        if oos_s["net_pnl"] < 0:
            wf_verdict += "  🔴 OOS net negative — edge may not be real.\n"
        elif drop > 5:
            wf_verdict += f"  ⚠️ {drop:.1f}% accuracy drop OOS — likely curve-fitted.\n"
        else:
            wf_verdict += "  ✅ OOS profitable — edge appears robust.\n"

    # ── Final report ──
    report = (
        f"🏛 *FINANCIAL PERFORMANCE AUDIT v2*\n"
        f"Capital: ₹{CONFIG['INITIAL_CASH']:,}  |  Period: {CONFIG['BACKTEST_PERIOD']}\n"
        f"Strategy: AdvancedRSI + EMA200  |  Costs: 0.4% exp + 0.1% slip/side\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 *IN-SAMPLE ({is_s['label']})*\n"
        f"{fmt_section(is_s)}"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🧪 *OUT-OF-SAMPLE ({oos_s['label']})*\n"
        f"{fmt_section(oos_s)}"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        f"{wf_verdict}\n"
        f"*── TIMEFRAME (combined) ──*\n{tf_lines}\n"
        f"*── STAR PERFORMER ──*\n"
        f"  {best['name']}  (+₹{best['profit']:,.0f})\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"_Losses set off vs gains same FY (Indian IT Act)._\n"
        f"_STCG @ 15% flat. Loss carry-forward: 8 yrs._\n"
        f"_Consult your CA before filing. Auto-generated._"
    )

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    send_msg(report)
    print("\n[Done]")
