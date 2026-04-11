"""
SOL/USDT — RSI(40) / WMA(15) Crossover Backtest
=================================================
• Data   : Binance public API, 5-minute candles, last 5 years
• Entry  : RSI crosses ABOVE WMA_RSI AND RSI < 60  → Long
• SL     : max(low of crossover candle, close - 1.3×ATR)
• TP     : close + 2.2 × (close - SL)
• Size   : fixed $1,000 USDC account, full capital per trade (no compounding)
• Report : sent to Telegram on completion
"""

import time
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="backtest.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
TOKEN   = "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg"
CHAT_ID = "1950462171"

SYMBOL     = "SOLUSDT"
INTERVAL   = "5m"
ACCOUNT    = 1_000.0      # USDC
RSI_PERIOD = 40
WMA_PERIOD = 15
ATR_PERIOD = 14
ATR_MULT   = 1.3          # SL = max(crossover low, close - ATR_MULT×ATR)
RR         = 2.2          # TP = entry + RR × (entry - SL)
RSI_MAX    = 60.0         # only enter if RSI < RSI_MAX at crossover

BINANCE_BASE = "https://api.binance.com"
LIMIT        = 1000       # max candles per request


# ── Telegram ─────────────────────────────────────────────────────────────────

def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
    except Exception as e:
        log.error(f"Telegram error: {e}")


def send_long_telegram(text: str) -> None:
    """Split messages that exceed Telegram's 4096-char limit."""
    chunk_size = 4000
    for i in range(0, len(text), chunk_size):
        send_telegram(text[i : i + chunk_size])
        time.sleep(0.5)


# ── Data fetch ───────────────────────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Paginate through Binance /api/v3/klines to fetch all candles
    between start_ms and end_ms (epoch milliseconds).
    """
    all_rows: list[list] = []
    current  = start_ms
    total_calls = 0

    print(f"⬇️  Fetching {symbol} {interval} data …  (this may take a few minutes)")
    send_telegram(f"⏳ *Backtest starting*\nFetching `{symbol}` `{interval}` data from Binance …")

    while current < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": current,
            "endTime":   end_ms,
            "limit":     LIMIT,
        }
        for attempt in range(1, 4):
            try:
                resp = requests.get(f"{BINANCE_BASE}/api/v3/klines",
                                    params=params, timeout=20)
                resp.raise_for_status()
                rows = resp.json()
                break
            except Exception as e:
                log.warning(f"Fetch attempt {attempt} failed: {e}")
                if attempt == 3:
                    raise
                time.sleep(5 * attempt)

        if not rows:
            break

        all_rows.extend(rows)
        current = rows[-1][6] + 1   # close-time of last candle + 1 ms
        total_calls += 1

        if total_calls % 50 == 0:
            pct = (current - start_ms) / (end_ms - start_ms) * 100
            print(f"   … {pct:.1f}% fetched ({len(all_rows):,} candles)")

        time.sleep(0.12)   # stay well within Binance rate limits

    df = pd.DataFrame(all_rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    print(f"✅ Downloaded {len(df):,} candles ({total_calls} API calls)")
    log.info(f"Downloaded {len(df):,} candles in {total_calls} calls")
    return df


# ── Indicators ───────────────────────────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# ── Backtest engine ──────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> dict:
    print("🔧 Calculating indicators …")
    df = df.copy()
    df["rsi"]     = calc_rsi(df["close"], RSI_PERIOD)
    df["wma_rsi"] = calc_wma(df["rsi"],   WMA_PERIOD)
    df["atr"]     = calc_atr(df,           ATR_PERIOD)
    df = df.dropna(subset=["rsi", "wma_rsi", "atr"]).reset_index(drop=True)

    trades: list[dict] = []
    in_trade   = False
    entry_px   = sl = tp = 0.0
    entry_time = None
    entry_idx  = 0

    print("🔁 Running trade simulation …")

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        # ── Manage open trade ──
        if in_trade:
            hit_sl = row["low"]  <= sl
            hit_tp = row["high"] >= tp

            if hit_sl and hit_tp:
                # Assume worst case: SL hit first
                outcome = "SL"
                exit_px = sl
            elif hit_sl:
                outcome = "SL"
                exit_px = sl
            elif hit_tp:
                outcome = "TP"
                exit_px = tp
            else:
                continue   # still in trade

            pnl_pct  = (exit_px - entry_px) / entry_px
            pnl_usdc = ACCOUNT * pnl_pct
            trades.append({
                "entry_time": entry_time,
                "exit_time":  row["close_time"],
                "entry":      entry_px,
                "exit":       exit_px,
                "sl":         sl,
                "tp":         tp,
                "outcome":    outcome,
                "pnl_usdc":   pnl_usdc,
                "pnl_pct":    pnl_pct * 100,
                "hold_bars":  i - entry_idx,
            })
            in_trade = False
            continue

        # ── Check for new entry signal ──
        bullish_cross = (prev["rsi"] <= prev["wma_rsi"]) and (row["rsi"] > row["wma_rsi"])
        rsi_below_60  = row["rsi"] < RSI_MAX

        if bullish_cross and rsi_below_60:
            entry_px   = row["close"]
            entry_time = row["open_time"]
            entry_idx  = i

            # SL = max(crossover candle low, close - 1.3×ATR)
            atr_sl  = entry_px - ATR_MULT * row["atr"]
            candle_sl = row["low"]
            sl = max(candle_sl, atr_sl)          # bigger distance = lower price? NO:
            # "bigger" means the one that is further from entry = lower price
            sl = min(candle_sl, atr_sl)           # SL must be BELOW entry
            # actually: we want the SL that is HIGHER (closer) to protect capital less
            # per spec: "whichever is bigger" → bigger distance from entry
            sl_atr_dist    = entry_px - atr_sl
            sl_candle_dist = entry_px - candle_sl
            if sl_atr_dist >= sl_candle_dist:
                sl = atr_sl
            else:
                sl = candle_sl

            risk = entry_px - sl
            if risk <= 0:
                continue   # degenerate candle, skip

            tp       = entry_px + RR * risk
            in_trade = True

    # ── Close any open trade at last bar ──
    if in_trade:
        last = df.iloc[-1]
        exit_px  = last["close"]
        pnl_pct  = (exit_px - entry_px) / entry_px
        trades.append({
            "entry_time": entry_time,
            "exit_time":  last["close_time"],
            "entry":      entry_px,
            "exit":       exit_px,
            "sl":         sl,
            "tp":         tp,
            "outcome":    "OPEN",
            "pnl_usdc":   ACCOUNT * pnl_pct,
            "pnl_pct":    pnl_pct * 100,
            "hold_bars":  len(df) - 1 - entry_idx,
        })

    return {"trades": trades, "df": df}


# ── Statistics ───────────────────────────────────────────────────────────────

def calc_stats(trades: list[dict], df: pd.DataFrame) -> dict:
    if not trades:
        return {"error": "No trades generated."}

    tdf = pd.DataFrame(trades)

    total       = len(tdf)
    wins        = (tdf["outcome"] == "TP").sum()
    losses      = (tdf["outcome"] == "SL").sum()
    open_t      = (tdf["outcome"] == "OPEN").sum()
    win_rate    = wins / total * 100

    gross_profit = tdf.loc[tdf["pnl_usdc"] > 0, "pnl_usdc"].sum()
    gross_loss   = tdf.loc[tdf["pnl_usdc"] < 0, "pnl_usdc"].sum()
    net_pnl      = tdf["pnl_usdc"].sum()
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")

    avg_win  = tdf.loc[tdf["pnl_usdc"] > 0, "pnl_usdc"].mean() if wins  else 0
    avg_loss = tdf.loc[tdf["pnl_usdc"] < 0, "pnl_usdc"].mean() if losses else 0
    avg_rr   = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Equity curve & drawdown
    equity = ACCOUNT + tdf["pnl_usdc"].cumsum()
    roll_max   = equity.cummax()
    drawdown   = (equity - roll_max) / roll_max * 100
    max_dd     = drawdown.min()
    max_dd_usdc = (equity - roll_max).min()

    # Sharpe (annualised, assume 5-min bars → 105,120 bars/year)
    bars_per_year = 105_120
    pnl_series    = tdf["pnl_usdc"]
    if pnl_series.std() > 0:
        sharpe = (pnl_series.mean() / pnl_series.std()) * math.sqrt(bars_per_year / (tdf["hold_bars"].mean() or 1))
    else:
        sharpe = 0.0

    # Sortino
    neg_returns = pnl_series[pnl_series < 0]
    downside_std = neg_returns.std() if len(neg_returns) > 1 else 1e-9
    sortino = (pnl_series.mean() / downside_std) * math.sqrt(bars_per_year / (tdf["hold_bars"].mean() or 1))

    # Calmar
    calmar = (net_pnl / ACCOUNT * 100) / abs(max_dd) if max_dd != 0 else float("inf")

    # Consecutive wins/losses
    streaks = tdf["outcome"].apply(lambda x: 1 if x == "TP" else (-1 if x == "SL" else 0))
    max_cons_win = max_cons_loss = cur = 0
    for s in streaks:
        if s == 1:
            cur = cur + 1 if cur > 0 else 1
            max_cons_win = max(max_cons_win, cur)
        elif s == -1:
            cur = cur - 1 if cur < 0 else -1
            max_cons_loss = min(max_cons_loss, cur)
        else:
            cur = 0

    avg_hold_h = tdf["hold_bars"].mean() * 5 / 60   # 5-min bars → hours

    data_start = df["open_time"].iloc[0]
    data_end   = df["open_time"].iloc[-1]

    return {
        "data_start":      data_start,
        "data_end":        data_end,
        "total_candles":   len(df),
        "total_trades":    total,
        "wins":            wins,
        "losses":          losses,
        "open_trades":     open_t,
        "win_rate":        win_rate,
        "net_pnl":         net_pnl,
        "gross_profit":    gross_profit,
        "gross_loss":      gross_loss,
        "profit_factor":   profit_factor,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "avg_rr":          avg_rr,
        "max_dd_pct":      max_dd,
        "max_dd_usdc":     max_dd_usdc,
        "sharpe":          sharpe,
        "sortino":         sortino,
        "calmar":          calmar,
        "max_cons_wins":   max_cons_win,
        "max_cons_losses": abs(max_cons_loss),
        "avg_hold_hours":  avg_hold_h,
        "final_equity":    ACCOUNT + net_pnl,
        "return_pct":      net_pnl / ACCOUNT * 100,
    }


def format_report(s: dict) -> str:
    if "error" in s:
        return f"❌ Backtest Error: {s['error']}"

    verdict = "✅ PROFITABLE" if s["net_pnl"] > 0 else "❌ UNPROFITABLE"

    return f"""
📊 *BACKTEST REPORT — SOLUSDT 5m*
{verdict}

*━━━━━━ DATA ━━━━━━*
📅 Period : {s['data_start'].strftime('%d %b %Y')} → {s['data_end'].strftime('%d %b %Y')}
🕯 Candles : {s['total_candles']:,}

*━━━━━━ STRATEGY ━━━━━━*
• Entry  : RSI({RSI_PERIOD}) crosses above WMA({WMA_PERIOD}) & RSI < {RSI_MAX}
• SL     : max(crossover low, close − {ATR_MULT}×ATR{ATR_PERIOD})
• TP     : entry + {RR}× risk
• Size   : $1,000 USDC fixed

*━━━━━━ TRADE STATS ━━━━━━*
📈 Total Trades : {s['total_trades']}
✅ Wins         : {s['wins']}
❌ Losses       : {s['losses']}
🔓 Still Open   : {s['open_trades']}
🎯 Win Rate     : {s['win_rate']:.2f}%
⏱ Avg Hold     : {s['avg_hold_hours']:.1f} hrs

*━━━━━━ P&L ━━━━━━*
💰 Net P&L      : ${s['net_pnl']:+.2f}
📈 Gross Profit : ${s['gross_profit']:.2f}
📉 Gross Loss   : ${s['gross_loss']:.2f}
🏦 Final Equity : ${s['final_equity']:.2f}
📊 Total Return : {s['return_pct']:+.2f}%
⚖️ Profit Factor: {s['profit_factor']:.2f}

*━━━━━━ RISK / REWARD ━━━━━━*
💚 Avg Win      : ${s['avg_win']:.2f}
❤️ Avg Loss     : ${s['avg_loss']:.2f}
📐 Avg R:R      : {s['avg_rr']:.2f}

*━━━━━━ DRAWDOWN ━━━━━━*
📉 Max Drawdown : {s['max_dd_pct']:.2f}%  (${s['max_dd_usdc']:.2f})

*━━━━━━ RATIOS ━━━━━━*
⚡ Sharpe Ratio  : {s['sharpe']:.3f}
🛡 Sortino Ratio : {s['sortino']:.3f}
🏔 Calmar Ratio  : {s['calmar']:.3f}

*━━━━━━ STREAKS ━━━━━━*
🔥 Max Cons. Wins   : {s['max_cons_wins']}
💀 Max Cons. Losses : {s['max_cons_losses']}
""".strip()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 5-year window in milliseconds
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - int(5 * 365.25 * 24 * 3600 * 1000)

    try:
        df = fetch_klines(SYMBOL, INTERVAL, start_ms, end_ms)

        result  = run_backtest(df)
        trades  = result["trades"]
        df_used = result["df"]

        stats   = calc_stats(trades, df_used)
        report  = format_report(stats)

        print("\n" + report)
        send_long_telegram(report)
        log.info("Backtest complete. Report sent.")

    except Exception as e:
        err = f"❌ Backtest crashed: {e}"
        print(err)
        log.exception("Backtest crashed")
        send_telegram(err)
