"""
SOL/USDT — RSI(40)/WMA(15) Crossover Backtest  v2  (Partial + Trail)
=====================================================================
EXIT PHASES
───────────
Phase 1  Entry → 1.5×R    : Full position, original SL
          SL hit           → EXIT ALL  (full loss)

Phase 2  At 1.5×R         : Book 50 %, move SL → entry + 0.3×R (lock-in)
          Locked SL hit    → EXIT remaining 50 %  (small profit on half)

Phase 3  At 2.2×R         : Book next 25 % (of original)
          Trail remaining 25 % with  SL = close − 1×ATR  (ratchet up only)
          Trail SL hit     → EXIT last 25 %

Account  : $1,000 USDC fixed per trade (position sized on full capital each trade)
"""

import time, math, requests, logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="backtest_v2.log", level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
TOKEN   = "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg"
CHAT_ID = "1950462171"

SYMBOL      = "SOLUSDT"
INTERVAL    = "5m"
ACCOUNT     = 1_000.0
RSI_PERIOD  = 40
WMA_PERIOD  = 15
ATR_PERIOD  = 14
ATR_MULT    = 1.3        # initial SL: close − 1.3×ATR
TRAIL_MULT  = 1.0        # phase-3 trail: close − 1.0×ATR
R1_MULT     = 1.5        # phase-2 trigger
R1_BOOK     = 0.50       # book 50 % at 1.5×R
LOCK_MULT   = 0.3        # SL moves to entry + 0.3×R after phase-2
R2_MULT     = 2.2        # phase-3 trigger
R2_BOOK     = 0.25       # book 25 % at 2.2×R (25 % remains for trail)
RSI_MAX     = 60.0
BINANCE_BASE = "https://api.binance.com"
LIMIT        = 1000


# ── Telegram ─────────────────────────────────────────────────────────────────
def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": text,
                                 "parse_mode": "Markdown"}, timeout=15)
    except Exception as e:
        log.error(f"Telegram: {e}")

def send_long(text: str) -> None:
    for i in range(0, len(text), 4000):
        send_telegram(text[i:i+4000])
        time.sleep(0.4)


# ── Data ─────────────────────────────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str,
                 start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows, current, calls = [], start_ms, 0
    print("⬇️  Fetching data …")
    send_telegram("⏳ *Backtest v2 starting*\nFetching `SOLUSDT` 5 m data …")

    while current < end_ms:
        params = dict(symbol=symbol, interval=interval,
                      startTime=current, endTime=end_ms, limit=LIMIT)
        for attempt in range(1, 4):
            try:
                r = requests.get(f"{BINANCE_BASE}/api/v3/klines",
                                 params=params, timeout=20)
                r.raise_for_status()
                rows = r.json()
                break
            except Exception as e:
                log.warning(f"Fetch attempt {attempt}: {e}")
                if attempt == 3: raise
                time.sleep(5 * attempt)

        if not rows: break
        all_rows.extend(rows)
        current = rows[-1][6] + 1
        calls  += 1
        if calls % 50 == 0:
            pct = (current - start_ms) / (end_ms - start_ms) * 100
            print(f"   … {pct:.1f}%  ({len(all_rows):,} candles)")
        time.sleep(0.12)

    df = pd.DataFrame(all_rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qvol","trades","tb","tq","ign"])
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = (df.drop_duplicates("open_time")
            .sort_values("open_time")
            .reset_index(drop=True))
    print(f"✅ {len(df):,} candles  ({calls} API calls)")
    return df


# ── Indicators ───────────────────────────────────────────────────────────────
def calc_rsi(close: pd.Series, p: int) -> pd.Series:
    d  = close.diff()
    ag = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    al = (-d).clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))

def calc_wma(s: pd.Series, p: int) -> pd.Series:
    w = np.arange(1, p+1, dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def calc_atr(df: pd.DataFrame, p: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-pc).abs(),
                    (df["low"] -pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p-1, min_periods=p).mean()


# ── Backtest ─────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame) -> dict:
    print("🔧 Calculating indicators …")
    df = df.copy()
    df["rsi"]     = calc_rsi(df["close"], RSI_PERIOD)
    df["wma_rsi"] = calc_wma(df["rsi"],   WMA_PERIOD)
    df["atr"]     = calc_atr(df,           ATR_PERIOD)
    df = df.dropna(subset=["rsi","wma_rsi","atr"]).reset_index(drop=True)

    trades: list[dict] = []

    # ── Trade state ──
    in_trade   = False
    phase      = 0          # 1 = full, 2 = 50% remain, 3 = 25% trail
    entry_px   = 0.0
    entry_time = None
    entry_idx  = 0
    sl         = 0.0
    r1_px      = 0.0        # price at 1.5×R
    r2_px      = 0.0        # price at 2.2×R
    lock_sl    = 0.0        # SL after phase-2
    trail_sl   = 0.0        # trailing SL in phase-3
    risk       = 0.0        # initial R
    pnl_booked = 0.0        # cumulative P&L already locked in this trade

    # exit-type counters (filled at trade close)
    ex_full_sl   = 0   # full position stopped before 1.5×R
    ex_lock_sl   = 0   # 50% stopped at lock SL (between 1.5R and 2.2R)
    ex_r1_book   = 0   # 50% exits at 1.5×R         (always paired with something)
    ex_r2_book   = 0   # 25% exits at 2.2×R
    ex_trail_sl  = 0   # 25% trail stopped

    print("🔁 Running simulation …")

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i-1]

        # ════════════════════════════════════════════════════════
        #  MANAGE OPEN TRADE
        # ════════════════════════════════════════════════════════
        if in_trade:
            hi, lo, cl = row["high"], row["low"], row["close"]
            atr_now    = row["atr"]

            # ── Phase 1: full position, original SL ──────────────
            if phase == 1:
                if lo <= sl:
                    # Full stop loss
                    pnl = -ACCOUNT * (risk / entry_px)          # full loss
                    trades.append(_make_trade(
                        entry_time, row["close_time"], entry_px, sl,
                        pnl + pnl_booked, i - entry_idx, "FULL_SL",
                        {"booked_1.5R": 0, "booked_2.2R": 0,
                         "trail_exit": 0, "exit_at": sl}
                    ))
                    ex_full_sl += 1
                    in_trade = False

                elif hi >= r1_px:
                    # ── Hit 1.5×R: book 50 %, move SL ──────────
                    book_pnl    = ACCOUNT * R1_BOOK * (r1_px - entry_px) / entry_px
                    pnl_booked += book_pnl
                    lock_sl     = entry_px + LOCK_MULT * risk
                    phase       = 2
                    ex_r1_book += 1
                    # don't close trade — fall through to phase-2 check same bar

            # ── Phase 2: 50 % remain, locked SL ─────────────────
            if phase == 2:
                if lo <= lock_sl:
                    # Locked SL on remaining 50 %
                    half_loss = ACCOUNT * (1 - R1_BOOK) * (lock_sl - entry_px) / entry_px
                    total_pnl = pnl_booked + half_loss
                    trades.append(_make_trade(
                        entry_time, row["close_time"], entry_px, lock_sl,
                        total_pnl, i - entry_idx, "LOCK_SL",
                        {"booked_1.5R": pnl_booked - half_loss,
                         "booked_2.2R": 0, "trail_exit": 0,
                         "exit_at": lock_sl}
                    ))
                    ex_lock_sl += 1
                    in_trade = False

                elif hi >= r2_px:
                    # ── Hit 2.2×R: book 25 %, start trail on 25 % ──
                    book_pnl    = ACCOUNT * R2_BOOK * (r2_px - entry_px) / entry_px
                    pnl_booked += book_pnl
                    trail_sl    = cl - TRAIL_MULT * atr_now   # init trail
                    phase       = 3
                    ex_r2_book += 1

            # ── Phase 3: 25 % remain, ATR trail ─────────────────
            if phase == 3:
                # Ratchet trail up only
                new_trail = cl - TRAIL_MULT * atr_now
                if new_trail > trail_sl:
                    trail_sl = new_trail

                if lo <= trail_sl:
                    trail_pnl  = ACCOUNT * (1 - R1_BOOK - R2_BOOK) * \
                                 (trail_sl - entry_px) / entry_px
                    total_pnl  = pnl_booked + trail_pnl
                    trades.append(_make_trade(
                        entry_time, row["close_time"], entry_px, trail_sl,
                        total_pnl, i - entry_idx, "TRAIL_SL",
                        {"booked_1.5R": pnl_booked - trail_pnl,
                         "booked_2.2R": "included",
                         "trail_exit":  trail_pnl,
                         "exit_at":     trail_sl}
                    ))
                    ex_trail_sl += 1
                    in_trade = False

            continue   # next candle

        # ════════════════════════════════════════════════════════
        #  ENTRY SIGNAL
        # ════════════════════════════════════════════════════════
        bull_cross = (prev["rsi"] <= prev["wma_rsi"]) and (row["rsi"] > row["wma_rsi"])
        if not (bull_cross and row["rsi"] < RSI_MAX):
            continue

        entry_px   = row["close"]
        entry_time = row["open_time"]
        entry_idx  = i

        # SL: larger distance wins (more conservative)
        atr_sl     = entry_px - ATR_MULT * row["atr"]
        candle_sl  = row["low"]
        sl         = atr_sl if (entry_px - atr_sl) >= (entry_px - candle_sl) else candle_sl

        risk = entry_px - sl
        if risk <= 0:
            continue

        r1_px      = entry_px + R1_MULT * risk
        r2_px      = entry_px + R2_MULT * risk
        pnl_booked = 0.0
        phase      = 1
        in_trade   = True

    # ── Close any open trade at last bar ──
    if in_trade:
        last = df.iloc[-1]
        exit_px = last["close"]
        rem     = 1 - R1_BOOK - R2_BOOK if phase == 3 else \
                  (1 - R1_BOOK if phase == 2 else 1.0)
        pnl     = pnl_booked + ACCOUNT * rem * (exit_px - entry_px) / entry_px
        trades.append(_make_trade(
            entry_time, last["close_time"], entry_px, exit_px,
            pnl, len(df)-1-entry_idx, f"OPEN_P{phase}", {}
        ))

    return {
        "trades": trades, "df": df,
        "counters": {
            "full_sl":  ex_full_sl,
            "lock_sl":  ex_lock_sl,
            "r1_book":  ex_r1_book,
            "r2_book":  ex_r2_book,
            "trail_sl": ex_trail_sl,
        }
    }


def _make_trade(entry_time, exit_time, entry, exit_px,
                pnl, hold_bars, exit_type, detail) -> dict:
    return {
        "entry_time": entry_time,
        "exit_time":  exit_time,
        "entry":      entry,
        "exit":       exit_px,
        "pnl_usdc":   pnl,
        "pnl_pct":    pnl / ACCOUNT * 100,
        "hold_bars":  hold_bars,
        "exit_type":  exit_type,
        **detail,
    }


# ── Statistics ───────────────────────────────────────────────────────────────
def calc_stats(trades: list[dict], df: pd.DataFrame, counters: dict) -> dict:
    if not trades:
        return {"error": "No trades generated."}

    tdf = pd.DataFrame(trades)
    total   = len(tdf)
    wins    = (tdf["pnl_usdc"] > 0).sum()
    losses  = (tdf["pnl_usdc"] < 0).sum()
    breakev = (tdf["pnl_usdc"] == 0).sum()

    gross_p = tdf.loc[tdf["pnl_usdc"] > 0, "pnl_usdc"].sum()
    gross_l = tdf.loc[tdf["pnl_usdc"] < 0, "pnl_usdc"].sum()
    net_pnl = tdf["pnl_usdc"].sum()
    pf      = gross_p / abs(gross_l) if gross_l else float("inf")

    avg_win  = tdf.loc[tdf["pnl_usdc"] > 0, "pnl_usdc"].mean() if wins   else 0
    avg_loss = tdf.loc[tdf["pnl_usdc"] < 0, "pnl_usdc"].mean() if losses else 0
    avg_rr   = abs(avg_win / avg_loss) if avg_loss else float("inf")

    equity   = ACCOUNT + tdf["pnl_usdc"].cumsum()
    roll_max = equity.cummax()
    dd_pct   = ((equity - roll_max) / roll_max * 100)
    max_dd   = dd_pct.min()
    max_dd_u = (equity - roll_max).min()

    bars_per_year = 105_120
    avg_hold = tdf["hold_bars"].mean() or 1
    std_pnl  = tdf["pnl_usdc"].std()
    sharpe   = (tdf["pnl_usdc"].mean() / std_pnl *
                math.sqrt(bars_per_year / avg_hold)) if std_pnl else 0

    neg      = tdf.loc[tdf["pnl_usdc"] < 0, "pnl_usdc"]
    dstd     = neg.std() if len(neg) > 1 else 1e-9
    sortino  = (tdf["pnl_usdc"].mean() / dstd *
                math.sqrt(bars_per_year / avg_hold))
    calmar   = (net_pnl / ACCOUNT * 100) / abs(max_dd) if max_dd else float("inf")

    # Streaks
    sign = tdf["pnl_usdc"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    mcw = mcl = cur = 0
    for s in sign:
        if s == 1:
            cur = cur+1 if cur > 0 else 1
            mcw = max(mcw, cur)
        elif s == -1:
            cur = cur-1 if cur < 0 else -1
            mcl = min(mcl, cur)
        else:
            cur = 0

    # Exit-type P&L breakdown
    for et in ["FULL_SL","LOCK_SL","TRAIL_SL"]:
        sub = tdf[tdf["exit_type"] == et]["pnl_usdc"]

    def _et(t):
        sub = tdf[tdf["exit_type"] == t]
        return len(sub), sub["pnl_usdc"].sum()

    return {
        "data_start":    df["open_time"].iloc[0],
        "data_end":      df["open_time"].iloc[-1],
        "total_candles": len(df),
        "total_trades":  total,
        "wins":          int(wins),
        "losses":        int(losses),
        "breakeven":     int(breakev),
        "win_rate":      wins / total * 100,
        "net_pnl":       net_pnl,
        "gross_profit":  gross_p,
        "gross_loss":    gross_l,
        "profit_factor": pf,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "avg_rr":        avg_rr,
        "max_dd_pct":    max_dd,
        "max_dd_usdc":   max_dd_u,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "calmar":        calmar,
        "max_cons_wins": mcw,
        "max_cons_loss": abs(mcl),
        "avg_hold_h":    avg_hold * 5 / 60,
        "final_equity":  ACCOUNT + net_pnl,
        "return_pct":    net_pnl / ACCOUNT * 100,
        # exit breakdown
        "n_full_sl":     counters["full_sl"],
        "n_lock_sl":     counters["lock_sl"],
        "n_r1_book":     counters["r1_book"],
        "n_r2_book":     counters["r2_book"],
        "n_trail_sl":    counters["trail_sl"],
        "pnl_full_sl":   _et("FULL_SL")[1],
        "pnl_lock_sl":   _et("LOCK_SL")[1],
        "pnl_trail_sl":  _et("TRAIL_SL")[1],
    }


def format_report(s: dict) -> str:
    if "error" in s:
        return f"❌ {s['error']}"

    verdict = "✅ PROFITABLE" if s["net_pnl"] > 0 else "❌ UNPROFITABLE"

    return f"""
📊 *BACKTEST v2 — SOLUSDT 5m  (Partial + Trail)*
{verdict}

*━━━━━━ DATA ━━━━━━*
📅 {s['data_start'].strftime('%d %b %Y')} → {s['data_end'].strftime('%d %b %Y')}
🕯 Candles : {s['total_candles']:,}

*━━━━━━ STRATEGY ━━━━━━*
• Entry  : RSI(40) × WMA(15), RSI < 60
• SL     : max(candle low, close − 1.3×ATR)
• Phase1 : full position until 1.5×R
• Phase2 : book 50 %, SL → entry + 0.3×R
• Phase3 : at 2.2×R book 25 %, trail 25 % at close − 1×ATR

*━━━━━━ TRADE STATS ━━━━━━*
📈 Total Trades  : {s['total_trades']}
✅ Profitable    : {s['wins']}
❌ Loss          : {s['losses']}
🟰 Breakeven     : {s['breakeven']}
🎯 Win Rate      : {s['win_rate']:.2f}%
⏱ Avg Hold      : {s['avg_hold_h']:.1f} hrs

*━━━━━━ EXIT BREAKDOWN ━━━━━━*
💀 Full SL  (pre-1.5R)    : {s['n_full_sl']} trades  →  ${s['pnl_full_sl']:+.2f}
🔒 Lock SL  (1.5R→2.2R)  : {s['n_lock_sl']} trades  →  ${s['pnl_lock_sl']:+.2f}
📌 Booked @ 1.5×R (50%)  : {s['n_r1_book']} times
📌 Booked @ 2.2×R (25%)  : {s['n_r2_book']} times
🎯 Trail SL (25% runner)  : {s['n_trail_sl']} trades  →  ${s['pnl_trail_sl']:+.2f}

*━━━━━━ P&L ━━━━━━*
💰 Net P&L       : ${s['net_pnl']:+.2f}
📈 Gross Profit  : ${s['gross_profit']:.2f}
📉 Gross Loss    : ${s['gross_loss']:.2f}
🏦 Final Equity  : ${s['final_equity']:.2f}
📊 Total Return  : {s['return_pct']:+.2f}%
⚖️ Profit Factor : {s['profit_factor']:.3f}

*━━━━━━ RISK / REWARD ━━━━━━*
💚 Avg Win       : ${s['avg_win']:.2f}
❤️ Avg Loss      : ${s['avg_loss']:.2f}
📐 Avg R:R       : {s['avg_rr']:.2f}

*━━━━━━ DRAWDOWN ━━━━━━*
📉 Max Drawdown  : {s['max_dd_pct']:.2f}%  (${s['max_dd_usdc']:.2f})

*━━━━━━ RATIOS ━━━━━━*
⚡ Sharpe        : {s['sharpe']:.3f}
🛡 Sortino       : {s['sortino']:.3f}
🏔 Calmar        : {s['calmar']:.3f}

*━━━━━━ STREAKS ━━━━━━*
🔥 Max Cons Wins   : {s['max_cons_wins']}
💀 Max Cons Losses : {s['max_cons_loss']}
""".strip()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - int(5 * 365.25 * 24 * 3600 * 1000)

    try:
        df     = fetch_klines(SYMBOL, INTERVAL, start_ms, end_ms)
        result = run_backtest(df)
        stats  = calc_stats(result["trades"], result["df"], result["counters"])
        report = format_report(stats)

        print("\n" + report)
        send_long(report)
        log.info("Backtest v2 complete.")

    except Exception as e:
        msg = f"❌ Backtest crashed: {e}"
        print(msg)
        log.exception("Crash")
        send_telegram(msg)
