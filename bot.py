"""
RSI(40) / WMA(15) Crossover Alert Bot
Symbols : MONUSDT, SOLUSDT  (Bybit Linear Perpetuals)
TFs     : 1h, 4h
Data    : Bybit V5 public API  – no API key needed
Alerts  : Telegram
"""

import time
import logging
import requests
import numpy as np
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg"
TELEGRAM_CHAT_ID   = "1950462171"

SYMBOLS     = ["MONUSDT", "SOLUSDT" ,"HYPEUSDT"]
TIMEFRAMES  = ["60", "240"]          # Bybit uses minutes: 60=1h, 240=4h
TF_LABELS   = {"60": "1h", "240": "4h"}

RSI_PERIOD   = 40
WMA_PERIOD   = 15
CANDLE_LIMIT = 300   # well above RSI(40)+WMA(15) warm-up requirement

POLL_INTERVAL = 900  # seconds (5 min is plenty for 1h/4h signals)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# STATE  –  (symbol, tf) -> "above" | "below"
# ─────────────────────────────────────────────
prev_state: dict = {}

# ─────────────────────────────────────────────
# BYBIT V5 KLINES
# ─────────────────────────────────────────────
BYBIT_BASE = "https://api.bybit.com"

def fetch_klines(symbol: str, interval: str, limit: int = CANDLE_LIMIT) -> list:
    """
    Fetch linear perpetual klines from Bybit V5.
    Returns closing prices oldest -> newest.
    Bybit max per request = 200, so we paginate if limit > 200.
    """
    url = f"{BYBIT_BASE}/v5/market/kline"
    all_closes = []        # list of (open_time_ms, close)
    end_time_ms = None

    per_call    = min(limit, 200)
    calls_needed = -(-limit // per_call)   # ceiling division

    for _ in range(calls_needed):
        params = {
            "category": "linear",
            "symbol":   symbol,
            "interval": interval,
            "limit":    per_call,
        }
        if end_time_ms is not None:
            params["end"] = end_time_ms

        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.error(f"Bybit kline fetch error {symbol} {interval}: {e}")
            break

        if data.get("retCode", -1) != 0:
            log.error(f"Bybit API error {symbol} {interval}: {data.get('retMsg')}")
            break

        rows = data["result"]["list"]
        # each row: [openTime, open, high, low, close, vol, turnover]
        if not rows:
            break

        for row in rows:
            all_closes.append((int(row[0]), float(row[4])))

        oldest_ts   = int(rows[-1][0])
        end_time_ms = oldest_ts - 1   # next page: fetch older candles

        if len(rows) < per_call:
            break   # no more history available

    if not all_closes:
        return []

    # Sort oldest -> newest, deduplicate
    all_closes.sort(key=lambda x: x[0])
    seen   = set()
    closes = []
    for ts, cls in all_closes:
        if ts not in seen:
            seen.add(ts)
            closes.append(cls)

    closes = closes[-limit:]   # keep most recent `limit` bars
    log.debug(f"Fetched {len(closes)} candles for {symbol} {interval}")
    return closes


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def calc_rsi_wilder(closes: list, period: int = RSI_PERIOD) -> list:
    """Wilder-smoothed RSI — matches TradingView default."""
    arr = np.array(closes, dtype=float)
    n   = len(arr)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi.tolist()

    deltas = np.diff(arr)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    rsi[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rsi[i]   = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return rsi.tolist()


def calc_wma(values: list, period: int = WMA_PERIOD) -> list:
    """Linearly Weighted MA — weight 1..period, highest on most recent bar."""
    arr     = np.array(values, dtype=float)
    n       = len(arr)
    wma     = np.full(n, np.nan)
    weights = np.arange(1, period + 1, dtype=float)
    denom   = weights.sum()

    for i in range(period - 1, n):
        seg = arr[i - period + 1 : i + 1]
        if not np.any(np.isnan(seg)):
            wma[i] = np.dot(seg, weights) / denom

    return wma.tolist()


# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────
def send_telegram(msg: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        log.info(f"TG sent | {msg[:80]}")
    except Exception as e:
        log.error(f"Telegram error: {e}")


# ─────────────────────────────────────────────
# SYMBOL VALIDATOR
# ─────────────────────────────────────────────
def validate_symbols(symbols: list) -> list:
    """Confirm each symbol exists as a Bybit linear perpetual."""
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/instruments-info",
            params={"category": "linear", "limit": 1000},
            timeout=10,
        )
        r.raise_for_status()
        valid = {s["symbol"] for s in r.json()["result"]["list"]}
    except Exception as e:
        log.error(f"Could not fetch Bybit instrument list: {e}")
        return symbols   # proceed anyway

    ok, bad = [], []
    for sym in symbols:
        if sym in valid:
            ok.append(sym)
        else:
            base  = sym.replace("USDT", "")
            close = sorted(v for v in valid if base in v)[:6]
            log.error(f"'{sym}' NOT on Bybit linear perpetuals. Closest matches: {close}")
            bad.append(sym)

    if bad:
        send_telegram(
            "⚠️ <b>Invalid Bybit symbol(s)</b>\n"
            + "\n".join(f"❌ {s}" for s in bad)
            + "\n\nUpdate SYMBOLS in the script."
        )
    return ok


# ─────────────────────────────────────────────
# CROSSOVER CHECK
# ─────────────────────────────────────────────
def check_crossover(symbol: str, tf: str) -> None:
    closes = fetch_klines(symbol, tf)
    min_required = RSI_PERIOD + WMA_PERIOD + 10

    if len(closes) < min_required:
        log.warning(
            f"Only {len(closes)} candles for {symbol} {TF_LABELS.get(tf,tf)} "
            f"(need {min_required}), skipping."
        )
        return

    rsi_vals = calc_rsi_wilder(closes, RSI_PERIOD)
    wma_vals = calc_wma(rsi_vals, WMA_PERIOD)

    # [-1] = forming candle (skip), [-2] = last closed, [-3] = previous closed
    rsi_curr = rsi_vals[-2]
    wma_curr = wma_vals[-2]

    # NaN guard
    if rsi_curr != rsi_curr or wma_curr != wma_curr:
        log.debug(f"NaN present {symbol} {TF_LABELS.get(tf,tf)} — still warming up")
        return

    tf_label = TF_LABELS.get(tf, tf)
    key      = (symbol, tf)
    curr_pos = "above" if rsi_curr > wma_curr else "below"

    if key not in prev_state:
        prev_state[key] = curr_pos
        log.info(
            f"[INIT] {symbol} {tf_label} | "
            f"RSI={rsi_curr:.2f} {'>' if curr_pos=='above' else '<'} WMA={wma_curr:.2f}"
        )
        return

    old_pos = prev_state[key]

    if old_pos == "below" and curr_pos == "above":
        prev_state[key] = curr_pos
        send_telegram(
            f"🟢 <b>BULLISH CROSSOVER</b>\n"
            f"📌 <b>{symbol}</b>  |  ⏱ <b>{tf_label}</b>\n"
            f"RSI(40) <b>{rsi_curr:.2f}</b> ↑ above WMA(15) <b>{wma_curr:.2f}</b>\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
        )
    elif old_pos == "above" and curr_pos == "below":
        prev_state[key] = curr_pos
        send_telegram(
            f"🔴 <b>BEARISH CROSSOVER</b>\n"
            f"📌 <b>{symbol}</b>  |  ⏱ <b>{tf_label}</b>\n"
            f"RSI(40) <b>{rsi_curr:.2f}</b> ↓ below WMA(15) <b>{wma_curr:.2f}</b>\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
        )
    else:
        log.debug(
            f"[NO CROSS] {symbol} {tf_label} | "
            f"RSI={rsi_curr:.2f} WMA={wma_curr:.2f} pos={curr_pos}"
        )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    log.info("=" * 55)
    log.info("  RSI(40)/WMA(15) Crossover Bot  |  Bybit Perpetuals")
    log.info(f"  Symbols    : {SYMBOLS}")
    log.info(f"  Timeframes : {[TF_LABELS[t] for t in TIMEFRAMES]}")
    log.info(f"  Poll every : {POLL_INTERVAL}s")
    log.info("=" * 55)

    active = validate_symbols(SYMBOLS)
    if not active:
        log.error("No valid symbols — exiting.")
        return

    send_telegram(
        "🤖 <b>RSI/WMA Alert Bot Started</b>  [Bybit]\n"
        f"Watching  : {', '.join(active)}\n"
        f"Timeframes: {', '.join(TF_LABELS[t] for t in TIMEFRAMES)}\n"
        f"Signal    : RSI(40) ✕ WMA(15)"
    )

    while True:
        for symbol in active:
            for tf in TIMEFRAMES:
                try:
                    check_crossover(symbol, tf)
                except Exception as e:
                    log.error(f"Unhandled error {symbol} {TF_LABELS.get(tf,tf)}: {e}")
                time.sleep(0.4)   # gentle rate limiting

        log.info(f"Cycle done. Sleeping {POLL_INTERVAL}s …")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

