"""
╔══════════════════════════════════════════════════════════════════╗
║   Renko ATR Strategy — LIVE FORWARD TESTER (Binance Public API)   ║
║   Same signal logic as the backtester. Sends Telegram alerts on   ║
║   entry (with SL/TP) and exit, and tracks running performance.    ║
╚══════════════════════════════════════════════════════════════════╝

WHAT THIS DOES
---------------
1. Polls Binance for the current price every POLL_INTERVAL_SEC seconds
   and checks any OPEN trade against its SL/TP in near-real-time.
2. Every time a new candle closes on your chosen timeframe, it pulls a
   fresh window of candles, recomputes ATR + the adaptive Renko bricks,
   and runs the *same* entry-signal logic as the backtester
   (consecutive-brick runs + duplicate-entry ATR-gap guard, one trade
   at a time).
3. Sends a Telegram message the moment a trade is opened (entry/SL/TP)
   and the moment it closes (TP/SL/manual), including the trade's
   net % result after fees+slippage.
4. Persists state to a local JSON file (forward_test_state.json) so
   the script can be killed/restarted without losing the open trade
   or trade history. That file is NOT a separate dashboard — it only
   exists so the bot can (a) survive restarts and (b) compute the
   win-rate/PF/expectancy numbers it reports back to you on Telegram.

RUNNING IT
----------
    pip install requests
    python forward_test_bot.py

Stop with Ctrl+C. It's safe to restart — it will reload open trade /
history from forward_test_state.json.
"""

import json
import os
import time
import traceback
from datetime import datetime, timezone

import requests

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "symbol"            : "SOLUSDT",
    "timeframe"         : "30m",       # any Binance interval: 1m,3m,5m,15m,30m,1h,4h,1d ...
    "context_candles"   : 1000,        # how many recent candles to keep in memory for ATR/Renko

    "atr_period"        : 14,
    "renko_mult"        : 1.0,         # brick size = renko_mult x ATR (re-anchored per new brick)
    "sl_mult"           : 1.5,         # SL distance = sl_mult x ATR
    "risk_reward_ratio" : 3.5,         # TP distance = risk_reward_ratio x SL distance
    "min_sell_bricks"   : 2,           # min consecutive bearish bricks before a LONG entry
    "min_buy_bricks"    : 2,           # min consecutive bullish bricks before a SHORT entry
    "atr_gap_mult"      : 1.0,         # duplicate-entry guard
    "allow_shorts"      : False,

    # execution assumptions (used only to compute the % result reported to you —
    # this script does NOT place real orders, it's forward-test / paper-trade only)
    "fee_pct"           : 0.02,
    "slippage_pct"      : 0.02,
    "exit_priority"     : "heuristic",   # "heuristic" / "SL" / "TP" for same-candle ambiguity

    "initial_capital"       : 1000.0,
    "position_sizing_mode"  : "risk_based",   # "risk_based" or "fixed_fraction"
    "risk_pct_per_trade"    : 2.0,
    "max_leverage"          : 3.0,

    # live-loop timing
    "poll_interval_sec"     : 15,      # how often to check price for SL/TP + look for a new closed candle
    "summary_every_n_trades": 5,       # send a performance summary every N closed trades (0 = never)

    # Telegram
    "telegram_bot_token": "8392707199:AAHjWHGLoZ3Udm4rS5JlgSaPLez1qZbHMOo",
    "telegram_chat_id"  : "1950462171",

    "state_file": "forward_test_state.json",
}

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE_URL  = "https://api.binance.com/api/v3/ticker/price"

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000,
}


# ─────────────────────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────────────────────
def send_telegram(cfg: dict, text: str):
    token = cfg["telegram_bot_token"]
    chat_id = cfg["telegram_chat_id"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, data={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=10)
        if not r.ok:
            print(f"[TELEGRAM] send failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"[TELEGRAM] send exception: {e}")


# ─────────────────────────────────────────────────────────────
#  STATE PERSISTENCE
# ─────────────────────────────────────────────────────────────
def default_state(cfg):
    return {
        "open_trade": None,          # dict or None
        "sell_run": 0,
        "buy_run": 0,
        "last_entry_price": 0.0,
        "processed_brick_key": None,     # (open_time, dir, close) of last brick we evaluated
        "last_processed_open_time": 0,   # candle open_time up to which bricks/ATR were considered
        "trade_history": [],         # closed trades, for performance stats
        "equity": cfg["initial_capital"],
    }


def load_state(cfg):
    path = cfg["state_file"]
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[STATE] Failed to load state file, starting fresh: {e}")
    return default_state(cfg)


def save_state(cfg, state):
    path = cfg["state_file"]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────────────────────
def fetch_recent_klines(symbol: str, interval: str, n: int):
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval '{interval}'")
    params = {"symbol": symbol, "interval": interval, "limit": n}
    r = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    r.raise_for_status()
    rows = r.json()

    now_ms = int(time.time() * 1000)
    # drop the still-open (in-progress) candle
    if rows and rows[-1][6] > now_ms:
        rows = rows[:-1]

    opens  = [float(c[1]) for c in rows]
    highs  = [float(c[2]) for c in rows]
    lows   = [float(c[3]) for c in rows]
    closes = [float(c[4]) for c in rows]
    times  = [int(c[0])   for c in rows]
    return opens, highs, lows, closes, times


def fetch_live_price(symbol: str) -> float:
    r = requests.get(BINANCE_PRICE_URL, params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])


# ─────────────────────────────────────────────────────────────
#  ATR (Wilder smoothing) — identical to the backtester
# ─────────────────────────────────────────────────────────────
def calc_atr(highs, lows, closes, period: int):
    n = len(closes)
    tr = [0.0] * n
    atr = [0.0] * n
    s = 0.0
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
        if i < period:
            s += tr[i]
        elif i == period:
            s += tr[i]
            atr[i] = s / period
        else:
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ─────────────────────────────────────────────────────────────
#  RENKO BUILDER — identical to the backtester (adaptive brick size)
# ─────────────────────────────────────────────────────────────
def build_renko(closes, times, atr_arr, mult: float):
    bricks = []
    ref = None
    ref_atr = None

    for i in range(len(closes)):
        a = atr_arr[i]
        if a == 0:
            continue
        if ref is None:
            ref = closes[i]
            ref_atr = a
            continue

        price = closes[i]
        brick_sz = ref_atr * mult

        while price >= ref + brick_sz:
            bricks.append({"dir": 1, "open": ref, "close": ref + brick_sz,
                            "idx": i, "open_time": times[i], "atr": ref_atr})
            ref += brick_sz
            ref_atr = a
            brick_sz = ref_atr * mult

        while price <= ref - brick_sz:
            bricks.append({"dir": -1, "open": ref, "close": ref - brick_sz,
                            "idx": i, "open_time": times[i], "atr": ref_atr})
            ref -= brick_sz
            ref_atr = a
            brick_sz = ref_atr * mult

    return bricks


def apply_fill_prices(side, entry, exit_price, slip_frac):
    if side == "LONG":
        entry_fill = entry * (1 + slip_frac)
        exit_fill = exit_price * (1 - slip_frac)
    else:
        entry_fill = entry * (1 - slip_frac)
        exit_fill = exit_price * (1 + slip_frac)
    return entry_fill, exit_fill


# ─────────────────────────────────────────────────────────────
#  ENTRY-SIGNAL PROCESSING (mirrors run_backtest's entry branch)
# ─────────────────────────────────────────────────────────────
def process_new_bricks(cfg, state, bricks):
    """
    Walk any bricks we haven't evaluated yet. If a trade is already open
    we don't evaluate signals (one trade at a time — matches backtester),
    but we still advance the "already seen" watermark so those bricks are
    skipped once the trade eventually closes (matches the backtester's
    critical fix: bricks formed while a trade was open must not be
    replayed as signals after exit).
    """
    allow_shorts = cfg.get("allow_shorts", False)
    new_trade_alert = None

    last_open_time = state["last_processed_open_time"]
    pending = [b for b in bricks if b["open_time"] >= last_open_time]
    # avoid reprocessing the exact brick(s) at the boundary
    if pending and state.get("_last_brick_fingerprint"):
        fp = state["_last_brick_fingerprint"]
        pending = [b for b in pending
                   if (b["open_time"], b["dir"], round(b["close"], 8)) != tuple(fp)]

    for b in pending:
        if state["open_trade"] is not None:
            # position already open — do not evaluate new entries,
            # just move the watermark forward
            state["last_processed_open_time"] = b["open_time"]
            state["_last_brick_fingerprint"] = [b["open_time"], b["dir"], round(b["close"], 8)]
            continue

        if b["dir"] == -1:
            if allow_shorts and state["buy_run"] >= cfg["min_buy_bricks"]:
                entry = b["close"]
                atr = b["atr"]
                dup = (state["last_entry_price"] > 0 and
                       abs(entry - state["last_entry_price"]) < cfg["atr_gap_mult"] * atr)
                if not dup:
                    sl_dist = cfg["sl_mult"] * atr
                    sl = entry + sl_dist
                    tp = entry - cfg["risk_reward_ratio"] * sl_dist
                    state["open_trade"] = {
                        "side": "SHORT", "entry": entry, "sl": sl, "tp": tp, "atr": atr,
                        "entry_time": b["open_time"],
                    }
                    new_trade_alert = dict(state["open_trade"])
            state["buy_run"] = 0
            state["sell_run"] += 1
        else:
            if state["sell_run"] >= cfg["min_sell_bricks"]:
                entry = b["close"]
                atr = b["atr"]
                dup = (state["last_entry_price"] > 0 and
                       abs(entry - state["last_entry_price"]) < cfg["atr_gap_mult"] * atr)
                if not dup:
                    sl_dist = cfg["sl_mult"] * atr
                    sl = entry - sl_dist
                    tp = entry + cfg["risk_reward_ratio"] * sl_dist
                    state["open_trade"] = {
                        "side": "LONG", "entry": entry, "sl": sl, "tp": tp, "atr": atr,
                        "entry_time": b["open_time"],
                    }
                    new_trade_alert = dict(state["open_trade"])
            state["sell_run"] = 0
            state["buy_run"] += 1

        state["last_processed_open_time"] = b["open_time"]
        state["_last_brick_fingerprint"] = [b["open_time"], b["dir"], round(b["close"], 8)]

    return new_trade_alert


# ─────────────────────────────────────────────────────────────
#  EXIT CHECK (real-time price vs SL/TP)
# ─────────────────────────────────────────────────────────────
def check_exit(cfg, state, price):
    t = state["open_trade"]
    if t is None:
        return None

    hit_tp = hit_sl = False
    if t["side"] == "LONG":
        hit_tp = price >= t["tp"]
        hit_sl = price <= t["sl"]
    else:
        hit_tp = price <= t["tp"]
        hit_sl = price >= t["sl"]

    if not (hit_tp or hit_sl):
        return None

    # if both somehow true in the same tick, treat SL as the conservative default
    outcome = "SL" if hit_sl else "TP"
    exit_price = t["sl"] if outcome == "SL" else t["tp"]

    fee_frac = cfg["fee_pct"] / 100.0
    slip_frac = cfg["slippage_pct"] / 100.0

    entry_fill, exit_fill = apply_fill_prices(t["side"], t["entry"], exit_price, slip_frac)
    if t["side"] == "LONG":
        gross_pct = (exit_fill - entry_fill) / entry_fill
    else:
        gross_pct = (entry_fill - exit_fill) / entry_fill
    net_pct = (gross_pct - 2 * fee_frac) * 100

    closed_trade = {
        "side": t["side"],
        "entry_time": datetime.fromtimestamp(t["entry_time"] / 1000, tz=timezone.utc).isoformat(),
        "exit_time": datetime.now(tz=timezone.utc).isoformat(),
        "entry": t["entry"],
        "sl": t["sl"],
        "tp": t["tp"],
        "atr": t["atr"],
        "outcome": outcome,
        "net_pct": net_pct,
    }

    # update virtual equity using the same sizing logic as the backtester
    mode = cfg.get("position_sizing_mode", "risk_based")
    risk_frac_cfg = cfg["risk_pct_per_trade"] / 100.0
    max_lev = cfg.get("max_leverage", 1.0)
    sl_dist_pct = abs(t["entry"] - t["sl"]) / t["entry"]
    if mode == "risk_based":
        position_fraction = 0.0 if sl_dist_pct <= 0 else min(risk_frac_cfg / sl_dist_pct, max_lev)
    else:
        position_fraction = min(risk_frac_cfg, max_lev)
    state["equity"] *= (1 + (net_pct / 100.0) * position_fraction)
    closed_trade["equity_after"] = state["equity"]

    state["trade_history"].append(closed_trade)
    state["last_entry_price"] = t["entry"]
    state["open_trade"] = None

    return closed_trade


# ─────────────────────────────────────────────────────────────
#  MESSAGE FORMATTING
# ─────────────────────────────────────────────────────────────
def fmt_entry_msg(cfg, trade):
    side_emoji = "🟢 LONG" if trade["side"] == "LONG" else "🔴 SHORT"
    return (
        f"<b>{side_emoji} — {cfg['symbol']} ({cfg['timeframe']})</b>\n"
        f"Entry: <code>{trade['entry']:.6g}</code>\n"
        f"SL:    <code>{trade['sl']:.6g}</code>\n"
        f"TP:    <code>{trade['tp']:.6g}</code>\n"
        f"ATR:   {trade['atr']:.6g}\n"
        f"Time:  {datetime.fromtimestamp(trade['entry_time']/1000, tz=timezone.utc):%Y-%m-%d %H:%M UTC}"
    )


def fmt_exit_msg(cfg, trade, state):
    emoji = "✅" if trade["outcome"] == "TP" else "❌"
    stats = compute_stats(state["trade_history"])
    return (
        f"<b>{emoji} {trade['outcome']} — {trade['side']} {cfg['symbol']}</b>\n"
        f"Net result: <b>{trade['net_pct']:+.2f}%</b>\n"
        f"Entry: <code>{trade['entry']:.6g}</code> → Exit: <code>{trade['sl'] if trade['outcome']=='SL' else trade['tp']:.6g}</code>\n"
        f"Equity: ${trade['equity_after']:.2f}\n\n"
        f"📊 Running stats ({stats['n_closed']} trades)\n"
        f"Win rate: {stats['win_rate']:.1f}% | PF: {stats['profit_factor']:.2f}\n"
        f"Expectancy: {stats['expectancy']:+.3f}% | Total return: {stats['total_return_pct']:+.1f}%"
    )


def compute_stats(history):
    n_closed = len(history)
    if n_closed == 0:
        return {"n_closed": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "expectancy": 0.0, "total_return_pct": 0.0}
    n_tp = sum(1 for t in history if t["outcome"] == "TP")
    win_rate = n_tp / n_closed * 100
    gains = sum(t["net_pct"] for t in history if t["net_pct"] > 0)
    losses = -sum(t["net_pct"] for t in history if t["net_pct"] < 0)
    profit_factor = gains / losses if losses > 0 else float("inf")
    expectancy = sum(t["net_pct"] for t in history) / n_closed
    initial = CONFIG["initial_capital"]
    final = history[-1]["equity_after"]
    total_return_pct = (final / initial - 1) * 100
    return {"n_closed": n_closed, "win_rate": win_rate, "profit_factor": profit_factor,
            "expectancy": expectancy, "total_return_pct": total_return_pct}


def fmt_summary_msg(cfg, state):
    stats = compute_stats(state["trade_history"])
    return (
        f"<b>📈 Performance summary — {cfg['symbol']} {cfg['timeframe']}</b>\n"
        f"Closed trades: {stats['n_closed']}\n"
        f"Win rate: {stats['win_rate']:.1f}%\n"
        f"Profit factor: {stats['profit_factor']:.2f}\n"
        f"Expectancy/trade: {stats['expectancy']:+.3f}%\n"
        f"Total return: {stats['total_return_pct']:+.1f}%\n"
        f"Equity: ${state['equity']:.2f}"
    )


# ─────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────
def main(cfg=None):
    cfg = cfg or CONFIG
    state = load_state(cfg)

    print(f"[BOOT] Forward tester starting for {cfg['symbol']} {cfg['timeframe']}")
    send_telegram(cfg, f"🚀 Forward tester started — {cfg['symbol']} {cfg['timeframe']}\n"
                        f"Open trade: {'yes' if state['open_trade'] else 'none'}\n"
                        f"Trades logged so far: {len(state['trade_history'])}")

    while True:
        try:
            # 1) real-time SL/TP check
            if state["open_trade"] is not None:
                price = fetch_live_price(cfg["symbol"])
                closed = check_exit(cfg, state, price)
                if closed:
                    send_telegram(cfg, fmt_exit_msg(cfg, closed, state))
                    save_state(cfg, state)

                    n = cfg.get("summary_every_n_trades", 0)
                    if n and len(state["trade_history"]) % n == 0:
                        send_telegram(cfg, fmt_summary_msg(cfg, state))

            # 2) look for new closed candles -> recompute ATR/Renko -> check for new signal
            opens, highs, lows, closes, times = fetch_recent_klines(
                cfg["symbol"], cfg["timeframe"], cfg["context_candles"])

            if len(closes) > cfg["atr_period"] + 1:
                atr_arr = calc_atr(highs, lows, closes, cfg["atr_period"])
                bricks = build_renko(closes, times, atr_arr, cfg["renko_mult"])

                new_trade = process_new_bricks(cfg, state, bricks)
                if new_trade:
                    send_telegram(cfg, fmt_entry_msg(cfg, new_trade))
                save_state(cfg, state)

            time.sleep(cfg["poll_interval_sec"])

        except KeyboardInterrupt:
            print("[STOP] Interrupted by user.")
            send_telegram(cfg, "🛑 Forward tester stopped (manual interrupt).")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
            time.sleep(min(60, cfg["poll_interval_sec"] * 4))


if __name__ == "__main__":
    main()
