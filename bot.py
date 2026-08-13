import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

import requests

CONFIG = {
    "symbol": "SOLUSDT",
    "timeframe": "30m",
    "seed_candles": 1000,
    "atr_period": 14,
    "renko_mult": 1.0,
    "sl_mult": 1.5,
    "risk_reward_ratio": 3.5,
    "min_sell_bricks": 2,
    "min_buy_bricks": 2,
    "atr_gap_mult": 1.0,
    "allow_shorts": False,
    "fee_pct": 0.02,
    "slippage_pct": 0.02,
    "exit_priority": "heuristic",
    "initial_capital": 1000.0,
    "position_sizing_mode": "risk_based",
    "risk_pct_per_trade": 2.0,
    "max_leverage": 3.0,
    "poll_interval_sec": 15,
    "summary_every_n_trades": 5,
    "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    "telegram_max_retries": 3,
    "state_file": "forward_test_state.json",
}

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000,
}

FINGERPRINT_FIELDS = [
    "symbol", "timeframe", "seed_candles", "atr_period", "renko_mult",
    "sl_mult", "risk_reward_ratio", "min_sell_bricks", "min_buy_bricks",
    "atr_gap_mult", "allow_shorts", "fee_pct", "slippage_pct",
    "exit_priority", "initial_capital", "position_sizing_mode",
    "risk_pct_per_trade", "max_leverage",
]


def now_ms():
    return int(time.time() * 1000)


def fp(cfg):
    payload = {k: cfg[k] for k in FINGERPRINT_FIELDS}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def atomic_save(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def default_state(cfg):
    return {
        "schema_version": 3,
        "strategy_fingerprint": fp(cfg),
        "warmed_up": False,
        "live_mode": False,
        "last_candle_open_time": 0,
        "last_candle_close": None,
        "atr_seed_sum": 0.0,
        "atr_seed_count": 0,
        "atr_value": 0.0,
        "renko_ref": None,
        "renko_ref_atr": None,
        "sell_run": 0,
        "buy_run": 0,
        "last_entry_price": 0.0,
        "open_trade": None,
        "trade_history": [],
        "equity": cfg["initial_capital"],
        "pending_notifications": [],
    }


def load_state(cfg):
    path = cfg["state_file"]
    if not os.path.exists(path):
        return default_state(cfg)

    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        print(f"[STATE] load failed: {e}")
        return default_state(cfg)

    if state.get("strategy_fingerprint") != fp(cfg):
        backup = f"{path}.{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.bak"
        try:
            os.replace(path, backup)
            print(f"[STATE] fingerprint changed; archived to {backup}")
        except OSError as e:
            print(f"[STATE] archive failed: {e}")
        return default_state(cfg)

    defaults = default_state(cfg)
    for k, v in defaults.items():
        state.setdefault(k, v)
    return state


def save_state(cfg, state):
    atomic_save(cfg["state_file"], state)


def telegram_once(cfg, text):
    token = cfg["telegram_bot_token"]
    chat_id = cfg["telegram_chat_id"]
    if not token or not chat_id:
        print("[TELEGRAM] missing bot token/chat id")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            data={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=(5, 15),
        )
        if r.ok:
            return True
        print(f"[TELEGRAM] {r.status_code}: {r.text[:300]}")
    except requests.RequestException as e:
        print(f"[TELEGRAM] request error: {e}")
    return False


def queue_message(state, text):
    state.setdefault("pending_notifications", []).append({
        "id": hashlib.sha256(f"{time.time_ns()}:{text}".encode()).hexdigest()[:20],
        "text": text,
        "attempts": 0,
    })


def send_telegram(cfg, state, text, queue_on_failure=True):
    retries = max(1, int(cfg.get("telegram_max_retries", 3)))
    for attempt in range(retries):
        if telegram_once(cfg, text):
            return True
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    if queue_on_failure:
        queue_message(state, text)
    return False


def flush_notifications(cfg, state):
    pending = state.get("pending_notifications", [])
    if not pending:
        return
    remaining = []
    for item in pending:
        if telegram_once(cfg, item.get("text", "")):
            continue
        item["attempts"] = int(item.get("attempts", 0)) + 1
        remaining.append(item)
    state["pending_notifications"] = remaining


def validate_interval(interval):
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported timeframe: {interval}")


def parse_rows(rows):
    return (
        [float(r[2]) for r in rows],
        [float(r[3]) for r in rows],
        [float(r[4]) for r in rows],
        [int(r[0]) for r in rows],
    )


def fetch_recent_closed(symbol, interval, n):
    validate_interval(interval)
    r = requests.get(
        BINANCE_KLINES_URL,
        params={"symbol": symbol, "interval": interval, "limit": n},
        timeout=(5, 15),
    )
    r.raise_for_status()
    rows = [row for row in r.json() if int(row[6]) <= now_ms()]
    return parse_rows(rows)


def fetch_closed_since(symbol, interval, last_open, limit=1000):
    validate_interval(interval)
    step = INTERVAL_MS[interval]
    cursor = last_open + step if last_open else 0
    current = now_ms()
    all_rows = []

    while cursor < current:
        r = requests.get(
            BINANCE_KLINES_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "limit": limit,
            },
            timeout=(5, 15),
        )
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break

        for row in rows:
            if int(row[0]) > last_open and int(row[6]) <= current:
                all_rows.append(row)

        next_cursor = int(rows[-1][0]) + step
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(rows) < limit:
            break
        time.sleep(0.05)

    unique = {int(r[0]): r for r in all_rows}
    rows = [unique[k] for k in sorted(unique)]
    return parse_rows(rows)


def fetch_forming(symbol, interval) -> Optional[dict]:
    validate_interval(interval)
    r = requests.get(
        BINANCE_KLINES_URL,
        params={"symbol": symbol, "interval": interval, "limit": 1},
        timeout=(5, 10),
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return None
    row = rows[-1]
    if int(row[6]) <= now_ms():
        return None
    return {
        "open_time": int(row[0]),
        "high": float(row[2]),
        "low": float(row[3]),
    }


def fetch_live_price(symbol):
    r = requests.get(
        BINANCE_PRICE_URL,
        params={"symbol": symbol},
        timeout=(5, 10),
    )
    r.raise_for_status()
    return float(r.json()["price"])


def process_candle(cfg, state, high, low, close, open_time):
    prev = state["last_candle_close"]
    state["last_candle_close"] = close

    if prev is None:
        return []

    tr = max(high - low, abs(high - prev), abs(low - prev))
    period = cfg["atr_period"]

    if state["atr_seed_count"] < period:
        state["atr_seed_sum"] += tr
        state["atr_seed_count"] += 1
        if state["atr_seed_count"] < period:
            return []
        state["atr_value"] = state["atr_seed_sum"] / period
    else:
        state["atr_value"] = (
            state["atr_value"] * (period - 1) + tr
        ) / period

    atr = state["atr_value"]
    if atr <= 0:
        return []

    if state["renko_ref"] is None:
        state["renko_ref"] = close
        state["renko_ref_atr"] = atr
        return []

    ref = state["renko_ref"]
    ref_atr = state["renko_ref_atr"]
    size = ref_atr * cfg["renko_mult"]
    bricks = []

    while close >= ref + size:
        c = ref + size
        bricks.append({"dir": 1, "open": ref, "close": c, "open_time": open_time, "atr": ref_atr})
        ref, ref_atr = c, atr
        size = ref_atr * cfg["renko_mult"]

    while close <= ref - size:
        c = ref - size
        bricks.append({"dir": -1, "open": ref, "close": c, "open_time": open_time, "atr": ref_atr})
        ref, ref_atr = c, atr
        size = ref_atr * cfg["renko_mult"]

    state["renko_ref"] = ref
    state["renko_ref_atr"] = ref_atr
    return bricks


def update_runs(state, bricks):
    for b in bricks:
        if b["dir"] == -1:
            state["sell_run"] += 1
            state["buy_run"] = 0
        else:
            state["buy_run"] += 1
            state["sell_run"] = 0


def seed_historical(cfg, state):
    highs, lows, closes, times = fetch_recent_closed(
        cfg["symbol"], cfg["timeframe"], cfg["seed_candles"]
    )
    if not times:
        raise RuntimeError("No closed candles available for seed")

    for h, l, c, t in zip(highs, lows, closes, times):
        bricks = process_candle(cfg, state, h, l, c, t)
        update_runs(state, bricks)
        state["last_candle_open_time"] = t

    state["warmed_up"] = True
    print(
        f"[SEED] {len(times)} candles | "
        f"ATR={state['atr_value']:.8f} | "
        f"RenkoRef={state['renko_ref']}"
    )


def apply_fill_prices(side, entry, exit_price, slip):
    if side == "LONG":
        return entry * (1 + slip), exit_price * (1 - slip)
    return entry * (1 - slip), exit_price * (1 + slip)


def make_entry(cfg, state, brick, fill_price):
    if state["open_trade"] is not None:
        return None

    atr = brick["atr"]
    ref = brick["close"]

    if brick["dir"] == -1:
        if not cfg["allow_shorts"] or state["buy_run"] < cfg["min_buy_bricks"]:
            return None
    else:
        if state["sell_run"] < cfg["min_sell_bricks"]:
            return None

    if (
        state["last_entry_price"] > 0
        and abs(ref - state["last_entry_price"]) < cfg["atr_gap_mult"] * atr
    ):
        return None

    entry = fill_price
    d = cfg["sl_mult"] * atr

    if brick["dir"] == 1:
        side = "LONG"
        sl = entry - d
        tp = entry + cfg["risk_reward_ratio"] * d
    else:
        side = "SHORT"
        sl = entry + d
        tp = entry - cfg["risk_reward_ratio"] * d

    trade = {
        "side": side,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "atr": atr,
        "entry_time": now_ms(),
        "signal_brick_close": ref,
        "post_entry_high": None,
        "post_entry_low": None,
    }
    state["open_trade"] = trade
    return dict(trade)


def process_live_bricks(cfg, state, bricks, fill_price):
    new_trade = None
    for b in bricks:
        if state["open_trade"] is None:
            candidate = make_entry(cfg, state, b, fill_price)
            if candidate:
                new_trade = candidate
        update_runs(state, [b])
    return new_trade


def resolve_ambiguous(cfg, trade):
    mode = cfg.get("exit_priority", "heuristic")
    if mode in ("SL", "TP"):
        return mode
    sl_d = abs(trade["entry"] - trade["sl"])
    tp_d = abs(trade["entry"] - trade["tp"])
    return "SL" if sl_d <= tp_d else "TP"


def close_trade(cfg, state, outcome, ambiguous):
    trade = state["open_trade"]
    exit_price = trade["sl"] if outcome == "SL" else trade["tp"]

    fee = cfg["fee_pct"] / 100.0
    slip = cfg["slippage_pct"] / 100.0
    entry_fill, exit_fill = apply_fill_prices(
        trade["side"], trade["entry"], exit_price, slip
    )

    if trade["side"] == "LONG":
        gross = (exit_fill - entry_fill) / entry_fill
    else:
        gross = (entry_fill - exit_fill) / entry_fill

    net_pct = (gross - 2 * fee) * 100

    risk_mode = cfg.get("position_sizing_mode", "risk_based")
    risk_frac = cfg["risk_pct_per_trade"] / 100.0
    max_lev = cfg.get("max_leverage", 1.0)

    sl_pct = abs(trade["entry"] - trade["sl"]) / trade["entry"]

    if risk_mode == "risk_based":
        position_fraction = 0.0 if sl_pct <= 0 else min(risk_frac / sl_pct, max_lev)
    else:
        position_fraction = min(risk_frac, max_lev)

    state["equity"] *= 1 + (net_pct / 100.0) * position_fraction

    closed = {
        "side": trade["side"],
        "entry_time": datetime.fromtimestamp(
            trade["entry_time"] / 1000, timezone.utc
        ).isoformat(),
        "exit_time": datetime.now(timezone.utc).isoformat(),
        "entry": trade["entry"],
        "sl": trade["sl"],
        "tp": trade["tp"],
        "atr": trade["atr"],
        "signal_brick_close": trade.get("signal_brick_close"),
        "entry_fill": entry_fill,
        "exit_fill": exit_fill,
        "outcome": outcome,
        "ambiguous_exit": ambiguous,
        "exit_priority_mode": cfg.get("exit_priority") if ambiguous else None,
        "net_pct": net_pct,
        "equity_after": state["equity"],
    }

    state["trade_history"].append(closed)
    state["last_entry_price"] = trade.get("signal_brick_close", trade["entry"])
    state["open_trade"] = None
    state["sell_run"] = 0
    state["buy_run"] = 0
    return closed


def check_exit(cfg, state, price, high=None, low=None):
    trade = state["open_trade"]
    if trade is None:
        return None

    eff_high = price if high is None else max(price, high)
    eff_low = price if low is None else min(price, low)

    if trade["side"] == "LONG":
        hit_tp = eff_high >= trade["tp"]
        hit_sl = eff_low <= trade["sl"]
    else:
        hit_tp = eff_low <= trade["tp"]
        hit_sl = eff_high >= trade["sl"]

    if not hit_tp and not hit_sl:
        return None

    ambiguous = hit_tp and hit_sl
    outcome = (
        resolve_ambiguous(cfg, trade)
        if ambiguous
        else ("TP" if hit_tp else "SL")
    )
    return close_trade(cfg, state, outcome, ambiguous)


def update_post_entry(state, high, low, candle_open_time):
    trade = state["open_trade"]
    if trade is None:
        return
    if candle_open_time <= trade["entry_time"]:
        return

    trade["post_entry_high"] = (
        high if trade["post_entry_high"] is None
        else max(trade["post_entry_high"], high)
    )
    trade["post_entry_low"] = (
        low if trade["post_entry_low"] is None
        else min(trade["post_entry_low"], low)
    )


def compute_stats(history, cfg):
    n = len(history)
    if n == 0:
        return {
            "n_closed": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "total_return_pct": 0.0
        }

    wins = sum(1 for t in history if t["outcome"] == "TP")
    win_rate = wins / n * 100

    gains = sum(t["net_pct"] for t in history if t["net_pct"] > 0)
    losses = -sum(t["net_pct"] for t in history if t["net_pct"] < 0)
    pf = gains / losses if losses > 0 else float("inf")
    expectancy = sum(t["net_pct"] for t in history) / n
    total_return = (history[-1]["equity_after"] / cfg["initial_capital"] - 1) * 100

    return {
        "n_closed": n,
        "win_rate": win_rate,
        "profit_factor": pf,
        "expectancy": expectancy,
        "total_return_pct": total_return,
    }


def fmt_entry(cfg, t):
    side = "🟢 LONG" if t["side"] == "LONG" else "🔴 SHORT"
    return (
        f"<b>{side} — {cfg['symbol']} ({cfg['timeframe']})</b>\n"
        f"Entry: <code>{t['entry']:.6g}</code>\n"
        f"SL: <code>{t['sl']:.6g}</code>\n"
        f"TP: <code>{t['tp']:.6g}</code>\n"
        f"ATR: {t['atr']:.6g}\n"
        f"Signal brick: <code>{t['signal_brick_close']:.6g}</code>\n"
        f"Time: {datetime.fromtimestamp(t['entry_time']/1000, timezone.utc):%Y-%m-%d %H:%M UTC}"
    )


def fmt_exit(cfg, t, state):
    stats = compute_stats(state["trade_history"], cfg)
    pf = "∞" if stats["profit_factor"] == float("inf") else f"{stats['profit_factor']:.2f}"
    note = ""
    if t.get("ambiguous_exit"):
        note = (
            "\n⚠️ <b>AMBIGUOUS EXIT</b>\n"
            f"Resolved as {t['outcome']} using "
            f"exit_priority='{t['exit_priority_mode']}'.\n"
            "Exact intra-poll order is unknown.\n"
        )
    return (
        f"<b>{'✅' if t['outcome']=='TP' else '❌'} "
        f"{t['outcome']} — {t['side']} {cfg['symbol']}</b>\n"
        f"Net: <b>{t['net_pct']:+.2f}%</b>\n"
        f"Entry: <code>{t['entry']:.6g}</code>\n"
        f"Exit: <code>{t['sl'] if t['outcome']=='SL' else t['tp']:.6g}</code>\n"
        f"Equity: ${t['equity_after']:.2f}\n"
        f"{note}\n"
        f"Trades: {stats['n_closed']}\n"
        f"Win rate: {stats['win_rate']:.1f}%\n"
        f"PF: {pf}\n"
        f"Expectancy: {stats['expectancy']:+.3f}%\n"
        f"Total return: {stats['total_return_pct']:+.1f}%"
    )


def fmt_summary(cfg, state):
    stats = compute_stats(state["trade_history"], cfg)
    pf = "∞" if stats["profit_factor"] == float("inf") else f"{stats['profit_factor']:.2f}"
    return (
        f"<b>📈 Performance summary — {cfg['symbol']} {cfg['timeframe']}</b>\n"
        f"Closed trades: {stats['n_closed']}\n"
        f"Win rate: {stats['win_rate']:.1f}%\n"
        f"Profit factor: {pf}\n"
        f"Expectancy/trade: {stats['expectancy']:+.3f}%\n"
        f"Total return: {stats['total_return_pct']:+.1f}%\n"
        f"Equity: ${state['equity']:.2f}"
    )


def main(cfg=None):
    cfg = cfg or CONFIG

    if not cfg["telegram_bot_token"]:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in the environment.")
    if not cfg["telegram_chat_id"]:
        raise RuntimeError("Set TELEGRAM_CHAT_ID in the environment.")

    state = load_state(cfg)

    if not state["warmed_up"]:
        print("[BOOT] Initial seed...")
        seed_historical(cfg, state)
        state["live_mode"] = False
        save_state(cfg, state)

        send_telegram(
            cfg,
            state,
            f"🚀 <b>Forward tester started</b>\n"
            f"{cfg['symbol']} ({cfg['timeframe']})\n"
            f"Seeded {cfg['seed_candles']} closed candles.\n"
            "Historical signals were not traded.\n"
            "Waiting for the next newly closed candle.",
        )
        save_state(cfg, state)
    else:
        print(
            f"[BOOT] Resuming | open="
            f"{'yes' if state['open_trade'] else 'no'} | "
            f"trades={len(state['trade_history'])}"
        )
        state["live_mode"] = False
        save_state(cfg, state)

        send_telegram(
            cfg,
            state,
            f"🔄 <b>Forward tester resumed</b>\n"
            f"{cfg['symbol']} ({cfg['timeframe']})\n"
            f"Open trade: {'yes' if state['open_trade'] else 'none'}\n"
            f"Closed trades: {len(state['trade_history'])}\n"
            "Missed signals will not be traded.",
        )
        save_state(cfg, state)

    while True:
        try:
            flush_notifications(cfg, state)

            highs, lows, closes, times = fetch_closed_since(
                cfg["symbol"],
                cfg["timeframe"],
                state["last_candle_open_time"],
            )

            if times:
                if not state["live_mode"]:
                    # Resume/bootstrap catch-up:
                    # update state, never open a missed trade.
                    for h, l, c, t in zip(highs, lows, closes, times):
                        bricks = process_candle(cfg, state, h, l, c, t)
                        state["last_candle_open_time"] = t

                        if state["open_trade"] is not None and t > state["open_trade"]["entry_time"]:
                            update_post_entry(state, h, l, t)
                            closed = check_exit(cfg, state, c, h, l)
                            if closed:
                                save_state(cfg, state)
                                send_telegram(cfg, state, fmt_exit(cfg, closed, state))
                                n = cfg["summary_every_n_trades"]
                                if n and len(state["trade_history"]) % n == 0:
                                    send_telegram(cfg, state, fmt_summary(cfg, state))

                        # Reconstruct signal-run state silently.
                        update_runs(state, bricks)

                    state["live_mode"] = True
                    save_state(cfg, state)

                else:
                    live_price = fetch_live_price(cfg["symbol"])

                    for h, l, c, t in zip(highs, lows, closes, times):
                        bricks = process_candle(cfg, state, h, l, c, t)
                        state["last_candle_open_time"] = t

                        if state["open_trade"] is not None and t > state["open_trade"]["entry_time"]:
                            update_post_entry(state, h, l, t)
                            closed = check_exit(cfg, state, c, h, l)
                            if closed:
                                save_state(cfg, state)
                                send_telegram(cfg, state, fmt_exit(cfg, closed, state))
                                n = cfg["summary_every_n_trades"]
                                if n and len(state["trade_history"]) % n == 0:
                                    send_telegram(cfg, state, fmt_summary(cfg, state))
                                # Don't enter from bricks on the same candle
                                update_runs(state, bricks)
                                continue

                        if state["open_trade"] is None:
                            new_trade = process_live_bricks(
                                cfg, state, bricks, live_price
                            )
                            if new_trade:
                                save_state(cfg, state)
                                send_telegram(cfg, state, fmt_entry(cfg, new_trade))
                        else:
                            update_runs(state, bricks)

            # Final current-price check.
            live_price = fetch_live_price(cfg["symbol"])

            if state["open_trade"] is not None:
                trade = state["open_trade"]
                high = trade["post_entry_high"]
                low = trade["post_entry_low"]

                forming = fetch_forming(cfg["symbol"], cfg["timeframe"])
                if forming and forming["open_time"] > trade["entry_time"]:
                    high = forming["high"] if high is None else max(high, forming["high"])
                    low = forming["low"] if low is None else min(low, forming["low"])

                closed = check_exit(
                    cfg,
                    state,
                    live_price,
                    high,
                    low,
                )
                if closed:
                    save_state(cfg, state)
                    send_telegram(cfg, state, fmt_exit(cfg, closed, state))
                    n = cfg["summary_every_n_trades"]
                    if n and len(state["trade_history"]) % n == 0:
                        send_telegram(cfg, state, fmt_summary(cfg, state))

            save_state(cfg, state)
            time.sleep(max(1, cfg["poll_interval_sec"]))

        except KeyboardInterrupt:
            print("[STOP] Manual interrupt.")
            send_telegram(cfg, state, "🛑 Forward tester stopped manually.")
            save_state(cfg, state)
            break

        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
            save_state(cfg, state)
            time.sleep(min(60, max(5, cfg["poll_interval_sec"] * 4)))


if __name__ == "__main__":
    main()
