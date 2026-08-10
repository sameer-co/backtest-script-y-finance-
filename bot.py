"""
╔══════════════════════════════════════════════════════════════════╗
║      SOL Renko ATR Strategy — 5m BACKTESTER (Binance public)     ║
║  Same signal logic as the forward tester, run over historical    ║
║  data. Produces a trade-by-trade CSV and sends a summary report  ║
║  + the CSV file to Telegram.                                     ║
╚══════════════════════════════════════════════════════════════════╝

Period       : 2025-08-10 -> 2026-08-10 (edit START_DATE / END_DATE)
Timeframe    : 5m
Symbol       : SOLUSDT
Cost model   : 0.05% on entry + 0.05% on exit  (0.1% roundtrip total)
Fill model   : entry at brick-close price, exit checked candle-by-candle
               against SL/TP using that candle's high/low. If a single
               candle's range contains BOTH the SL and TP level, SL is
               assumed to fill first (conservative -- OHLC data alone
               can't tell you the true intra-candle path).

Only one trade is open at a time (matches the live forward tester).
Requires internet access to api.binance.com and api.telegram.org.
"""

import csv
import time
import requests
import numpy as np
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────
#  CONFIG — edit these
# ─────────────────────────────────────────────────────────────
TG_TOKEN   = "8661081060:AAGtNViZMS6FSl_7vQeMz1TcCnzrFddu7z4"   # <-- rotate this, it was pasted in chat
TG_CHAT_ID = "1950462171"
TG_URL     = f"https://api.telegram.org/bot{TG_TOKEN}"

SYMBOL      = "SOLUSDT"
INTERVAL    = "5m"
START_DATE  = "2025-08-10"   # UTC, inclusive
END_DATE    = "2026-08-10"   # UTC, exclusive (today's date -> pulls up to now)

SETTINGS = {
    "atr_period"      : 14,
    "renko_mult"      : 1.0,     # brick size = renko_mult x ATR
    "sl_mult"         : 1.5,     # SL = sl_mult x ATR below entry
    "tp_mult"         : 3.0,     # TP = tp_mult x SL above entry
    "min_sell_bricks" : 2,       # consecutive bearish bricks required before a buy
}

ENTRY_FEE_PCT = 0.0005   # 0.05%
EXIT_FEE_PCT  = 0.0005   # 0.05%   -> 0.10% roundtrip total

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
OUT_CSV = "sol_renko_5m_backtest_trades.csv"

SL_FILLS_FIRST_ON_SAME_CANDLE = True  # conservative assumption, see docstring


# ─────────────────────────────────────────────────────────────
#  DATA FETCH — paginate Binance klines across the full range
# ─────────────────────────────────────────────────────────────
def date_to_ms(d: str) -> int:
    return int(datetime.strptime(d, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines_range(symbol: str, interval: str, start_ms: int, end_ms: int):
    """Paginate /klines with startTime, 1000 candles per call."""
    all_rows = []
    cursor = start_ms
    interval_ms = 5 * 60 * 1000  # 5m in ms

    while cursor < end_ms:
        params = {
            "symbol"   : symbol,
            "interval" : interval,
            "startTime": cursor,
            "endTime"  : end_ms,
            "limit"    : 1000,
        }
        for attempt in range(5):
            try:
                r = requests.get(BINANCE_KLINES, params=params, timeout=15)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                print(f"[fetch] retry {attempt+1}/5: {e}")
                time.sleep(2)
        else:
            raise RuntimeError("Failed to fetch klines after 5 retries")

        if not batch:
            break

        all_rows.extend(batch)
        last_open_time = batch[-1][0]
        cursor = last_open_time + interval_ms

        print(f"[fetch] {len(all_rows)} candles so far "
              f"(up to {datetime.fromtimestamp(last_open_time/1000, tz=timezone.utc)})")

        if len(batch) < 1000:
            break

        time.sleep(0.25)  # be polite to the public rate limit

    return all_rows


def load_data(symbol, interval, start_date, end_date):
    start_ms = date_to_ms(start_date)
    end_ms   = date_to_ms(end_date)
    raw = fetch_klines_range(symbol, interval, start_ms, end_ms)
    raw = [c for c in raw if c[0] < end_ms]   # drop anything at/after end

    opens  = np.array([float(c[1]) for c in raw])
    highs  = np.array([float(c[2]) for c in raw])
    lows   = np.array([float(c[3]) for c in raw])
    closes = np.array([float(c[4]) for c in raw])
    times  = np.array([int(c[0])   for c in raw])
    return opens, highs, lows, closes, times


# ─────────────────────────────────────────────────────────────
#  ATR (Wilder) — identical to forward tester
# ─────────────────────────────────────────────────────────────
def calc_atr(highs, lows, closes, period: int) -> np.ndarray:
    n   = len(closes)
    tr  = np.zeros(n)
    atr = np.zeros(n)
    s   = 0.0
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i]  - closes[i-1]))
        if i < period:
            s += tr[i]
        elif i == period:
            s += tr[i]
            atr[i] = s / period
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


# ─────────────────────────────────────────────────────────────
#  RENKO BUILDER — identical to forward tester
# ─────────────────────────────────────────────────────────────
def build_renko(closes, atr_arr, mult: float):
    bricks  = []
    ref     = None
    ref_atr = None

    for i in range(len(closes)):
        a = atr_arr[i]
        if a == 0:
            continue
        if ref is None:
            ref     = closes[i]
            ref_atr = a
            continue

        price    = closes[i]
        brick_sz = ref_atr * mult

        while price >= ref + brick_sz:
            bricks.append({"dir": 1, "open": ref, "close": ref + brick_sz,
                           "idx": i, "atr": ref_atr})
            ref     += brick_sz
            ref_atr  = a
            brick_sz = ref_atr * mult

        while price <= ref - brick_sz:
            bricks.append({"dir": -1, "open": ref, "close": ref - brick_sz,
                           "idx": i, "atr": ref_atr})
            ref     -= brick_sz
            ref_atr  = a
            brick_sz = ref_atr * mult

    return bricks


# ─────────────────────────────────────────────────────────────
#  SIGNAL SCAN — same rule as forward tester: bullish brick after
#  >= min_sell_bricks bearish bricks in a row = BUY
# ─────────────────────────────────────────────────────────────
def find_all_signals(bricks, min_sell_bricks, sl_mult, tp_mult):
    signals = []
    sell_run = 0
    for b in bricks:
        if b["dir"] == -1:
            sell_run += 1
        else:
            if sell_run >= min_sell_bricks:
                entry = b["close"]
                atr   = b["atr"]
                sl    = entry - sl_mult * atr
                tp    = entry + tp_mult * sl_mult * atr
                signals.append({
                    "idx"      : b["idx"],
                    "entry"    : entry,
                    "sl"       : sl,
                    "tp"       : tp,
                    "atr"      : atr,
                    "sell_run" : sell_run,
                })
            sell_run = 0
    return signals


# ─────────────────────────────────────────────────────────────
#  TRADE SIMULATION — one position at a time, walk forward
#  candle-by-candle from the entry candle checking SL/TP
# ─────────────────────────────────────────────────────────────
def simulate_trades(signals, highs, lows, times, n_candles):
    trades = []
    cursor_idx = -1   # index up to which we're "in a trade" / blocked

    for sig in signals:
        if sig["idx"] <= cursor_idx:
            continue  # would have been in a trade already, skip like live bot

        entry_idx = sig["idx"]
        entry     = sig["entry"]
        sl        = sig["sl"]
        tp        = sig["tp"]

        exit_type  = "OPEN"
        exit_price = None
        exit_idx   = None

        for j in range(entry_idx + 1, n_candles):
            hit_tp = highs[j] >= tp
            hit_sl = lows[j]  <= sl

            if hit_tp and hit_sl:
                if SL_FILLS_FIRST_ON_SAME_CANDLE:
                    exit_type, exit_price = "SL", sl
                else:
                    exit_type, exit_price = "TP", tp
            elif hit_sl:
                exit_type, exit_price = "SL", sl
            elif hit_tp:
                exit_type, exit_price = "TP", tp
            else:
                continue

            exit_idx = j
            break

        gross_pct = (exit_price - entry) / entry if exit_price else None
        net_pct   = (gross_pct - ENTRY_FEE_PCT - EXIT_FEE_PCT) if gross_pct is not None else None

        trades.append({
            "entry_time" : datetime.fromtimestamp(times[entry_idx]/1000, tz=timezone.utc),
            "entry_price": entry,
            "sl"         : sl,
            "tp"         : tp,
            "atr"        : sig["atr"],
            "sell_bricks_before": sig["sell_run"],
            "exit_time"  : (datetime.fromtimestamp(times[exit_idx]/1000, tz=timezone.utc)
                             if exit_idx is not None else None),
            "exit_price" : exit_price,
            "exit_type"  : exit_type,
            "gross_pnl_pct": round(gross_pct * 100, 4) if gross_pct is not None else None,
            "net_pnl_pct"  : round(net_pct * 100, 4) if net_pct is not None else None,
        })

        # block further signals until this trade closes (or data ends)
        cursor_idx = exit_idx if exit_idx is not None else n_candles

    return trades


# ─────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────
def compute_metrics(trades):
    closed = [t for t in trades if t["exit_type"] in ("TP", "SL")]
    wins   = [t for t in closed if t["exit_type"] == "TP"]
    losses = [t for t in closed if t["exit_type"] == "SL"]
    open_t = [t for t in trades if t["exit_type"] == "OPEN"]

    total = len(closed)
    n_win = len(wins)
    n_loss = len(losses)
    winrate = (n_win / total * 100) if total else 0.0

    avg_win_pct  = np.mean([t["net_pnl_pct"] for t in wins])  if wins  else 0.0
    avg_loss_pct = np.mean([t["net_pnl_pct"] for t in losses]) if losses else 0.0
    ev_pct = (winrate/100 * avg_win_pct) + ((1 - winrate/100) * avg_loss_pct)

    gross_win  = sum(t["net_pnl_pct"] for t in wins)   if wins else 0.0
    gross_loss = abs(sum(t["net_pnl_pct"] for t in losses)) if losses else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    # simple compounding equity curve (1 unit, full size, no leverage)
    equity = 1.0
    for t in closed:
        equity *= (1 + t["net_pnl_pct"] / 100)
    total_return_pct = (equity - 1) * 100

    return {
        "total_trades"   : total,
        "wins"           : n_win,
        "losses"         : n_loss,
        "still_open"     : len(open_t),
        "winrate_pct"    : round(winrate, 2),
        "avg_win_pct"    : round(avg_win_pct, 3),
        "avg_loss_pct"   : round(avg_loss_pct, 3),
        "ev_pct_per_trade": round(ev_pct, 4),
        "profit_factor"  : round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "compounded_return_pct": round(total_return_pct, 2),
    }


# ─────────────────────────────────────────────────────────────
#  CSV OUTPUT
# ─────────────────────────────────────────────────────────────
def write_csv(trades, path):
    fields = ["entry_time", "entry_price", "sl", "tp", "atr", "sell_bricks_before",
              "exit_time", "exit_price", "exit_type", "gross_pnl_pct", "net_pnl_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in trades:
            w.writerow(t)


# ─────────────────────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────────────────────
def send_tg_message(text, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(f"{TG_URL}/sendMessage", json={
                "chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"
            }, timeout=10)
            if r.status_code == 200:
                return True
            print(f"[TG msg] HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[TG msg] error attempt {attempt+1}: {e}")
        time.sleep(2)
    return False


def send_tg_document(path, caption="", retries=3):
    for attempt in range(retries):
        try:
            with open(path, "rb") as f:
                r = requests.post(f"{TG_URL}/sendDocument",
                                   data={"chat_id": TG_CHAT_ID, "caption": caption},
                                   files={"document": f}, timeout=30)
            if r.status_code == 200:
                return True
            print(f"[TG doc] HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[TG doc] error attempt {attempt+1}: {e}")
        time.sleep(2)
    return False


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print(f"Fetching {SYMBOL} {INTERVAL} candles from {START_DATE} to {END_DATE} ...")
    opens, highs, lows, closes, times = load_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    n = len(closes)
    print(f"Loaded {n} candles.")
    if n < SETTINGS["atr_period"] + 5:
        print("Not enough data.")
        return

    atr_arr = calc_atr(highs, lows, closes, SETTINGS["atr_period"])
    bricks  = build_renko(closes, atr_arr, SETTINGS["renko_mult"])
    print(f"Built {len(bricks)} Renko bricks.")

    signals = find_all_signals(bricks, SETTINGS["min_sell_bricks"],
                                SETTINGS["sl_mult"], SETTINGS["tp_mult"])
    print(f"Found {len(signals)} raw signals.")

    trades = simulate_trades(signals, highs, lows, times, n)
    metrics = compute_metrics(trades)

    write_csv(trades, OUT_CSV)
    print(f"Wrote {len(trades)} trades to {OUT_CSV}")

    summary = (
        f"📊 <b>SOL Renko ATR Backtest — 5m</b>\n"
        f"Symbol   : <code>{SYMBOL}</code>\n"
        f"Period   : {START_DATE} → {END_DATE}\n"
        f"ATR({SETTINGS['atr_period']}) | Brick {SETTINGS['renko_mult']}×ATR | "
        f"SL {SETTINGS['sl_mult']}×ATR | TP {SETTINGS['tp_mult']}×SL | "
        f"Min sell bricks {SETTINGS['min_sell_bricks']}\n"
        f"Roundtrip cost: {(ENTRY_FEE_PCT+EXIT_FEE_PCT)*100:.2f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Total trades : {metrics['total_trades']}  "
        f"(open at end: {metrics['still_open']})\n"
        f"Wins / Losses: {metrics['wins']} / {metrics['losses']}\n"
        f"Win rate     : {metrics['winrate_pct']}%\n"
        f"Avg win      : {metrics['avg_win_pct']}%\n"
        f"Avg loss     : {metrics['avg_loss_pct']}%\n"
        f"EV / trade   : {metrics['ev_pct_per_trade']}%\n"
        f"Profit factor: {metrics['profit_factor']}\n"
        f"Compounded return: {metrics['compounded_return_pct']}%\n"
        f"⏰ Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    print(summary.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", ""))

    send_tg_message(summary)
    send_tg_document(OUT_CSV, caption=f"{SYMBOL} 5m Renko backtest trades "
                                       f"({START_DATE} to {END_DATE})")


if __name__ == "__main__":
    main()
