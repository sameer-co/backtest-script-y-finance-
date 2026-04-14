TOKEN   = "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg"
CHAT_ID = "1950462171"
"""
RSI(40) / WMA(15) Crossover Alert Bot
Symbols : MONOUSDT, SOLUSDT
Timeframes: 1h, 4h
Exchange : Binance Futures (public klines – no API key needed)
Alerts   : Telegram
"""

import time
import logging
import requests
import numpy as np
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CONFIG – fill these in before running
# ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg"
TELEGRAM_CHAT_ID   = "1950462171"

SYMBOLS     = ["MONOUSDT", "SOLUSDT"]
TIMEFRAMES  = ["1h", "4h"]

RSI_PERIOD  = 40
WMA_PERIOD  = 15

# How many candles to fetch (must be > RSI_PERIOD + WMA_PERIOD + buffer)
# 300 gives plenty of warm-up for both RSI(40) and WMA(15)
CANDLE_LIMIT = 400

# Poll interval in seconds (60 s = check once per minute)
POLL_INTERVAL = 900

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
# STATE  – tracks previous crossover direction
# key : (symbol, tf)  value: "above" | "below" | None
# ─────────────────────────────────────────────
prev_state: dict = {}


# ─────────────────────────────────────────────
# BINANCE FUTURES KLINES
# ─────────────────────────────────────────────
BINANCE_FUTURES_BASE = "https://fapi.binance.com"

def fetch_klines(symbol: str, interval: str, limit: int = CANDLE_LIMIT) -> list[float]:
    """Return list of closing prices (oldest → newest)."""
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        closes = [float(c[4]) for c in r.json()]   # index 4 = close price
        log.debug(f"Fetched {len(closes)} candles for {symbol} {interval}")
        return closes
    except Exception as e:
        log.error(f"Kline fetch failed {symbol} {interval}: {e}")
        return []


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def calc_rsi_wilder(closes: list[float], period: int = RSI_PERIOD) -> list[float]:
    """
    Wilder-smoothed RSI.
    Returns array aligned with closes (NaN for warm-up bars).
    """
    closes_arr = np.array(closes, dtype=float)
    n = len(closes_arr)
    rsi = np.full(n, np.nan)

    if n < period + 1:
        return rsi.tolist()

    deltas = np.diff(closes_arr)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # First average (simple mean of first `period` changes)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    # RSI at the `period`-th bar
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder smoothing for remaining bars
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi.tolist()


def calc_wma(values: list[float], period: int = WMA_PERIOD) -> list[float]:
    """
    Weighted Moving Average.
    Weights: 1, 2, 3 … period  (most recent = highest weight).
    Returns array aligned with values (NaN for warm-up bars).
    """
    arr = np.array(values, dtype=float)
    n   = len(arr)
    wma = np.full(n, np.nan)
    weights = np.arange(1, period + 1, dtype=float)
    denom   = weights.sum()

    for i in range(period - 1, n):
        segment = arr[i - period + 1 : i + 1]
        if np.any(np.isnan(segment)):
            continue
        wma[i] = np.dot(segment, weights) / denom

    return wma.tolist()


# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────
def send_telegram(msg: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        log.info(f"TG sent: {msg[:80]}")
    except Exception as e:
        log.error(f"Telegram send failed: {e}")


# ─────────────────────────────────────────────
# CROSSOVER CHECK
# ─────────────────────────────────────────────
def check_crossover(symbol: str, tf: str) -> None:
    closes = fetch_klines(symbol, tf)
    if len(closes) < CANDLE_LIMIT // 2:
        log.warning(f"Not enough data for {symbol} {tf}, skipping.")
        return

    rsi_vals = calc_rsi_wilder(closes, RSI_PERIOD)
    wma_vals = calc_wma(rsi_vals, WMA_PERIOD)

    # We need the last two completed bars (index -2 and -1)
    # (index -1 is the *current* forming candle – use -2 as confirmed, -3 as previous)
    # Use bars [-3] and [-2] to detect a confirmed crossover on the last closed candle.
    try:
        rsi_prev = rsi_vals[-3]
        rsi_curr = rsi_vals[-2]
        wma_prev = wma_vals[-3]
        wma_curr = wma_vals[-2]
    except IndexError:
        log.warning(f"Index error on {symbol} {tf}")
        return

    if any(v != v for v in [rsi_prev, rsi_curr, wma_prev, wma_curr]):  # NaN check
        log.debug(f"NaN values present for {symbol} {tf}, still in warm-up.")
        return

    key = (symbol, tf)
    current_state = "above" if rsi_curr > wma_curr else "below"

    if key not in prev_state:
        prev_state[key] = current_state
        log.info(f"Init state {symbol} {tf}: RSI {rsi_curr:.2f} {'>' if current_state=='above' else '<'} WMA {wma_curr:.2f}")
        return

    old_state = prev_state[key]

    if old_state == "below" and current_state == "above":
        # ✅ Bullish crossover – RSI crossed above WMA
        prev_state[key] = current_state
        msg = (
            f"🟢 <b>BULLISH CROSSOVER</b>\n"
            f"📌 Symbol : <b>{symbol}</b>\n"
            f"⏱ TF     : <b>{tf}</b>\n"
            f"📈 RSI(40): <b>{rsi_curr:.2f}</b> crossed ↑ WMA(15): <b>{wma_curr:.2f}</b>\n"
            f"🕐 UTC    : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
        )
        send_telegram(msg)

    elif old_state == "above" and current_state == "below":
        # 🔴 Bearish crossover – RSI crossed below WMA
        prev_state[key] = current_state
        msg = (
            f"🔴 <b>BEARISH CROSSOVER</b>\n"
            f"📌 Symbol : <b>{symbol}</b>\n"
            f"⏱ TF     : <b>{tf}</b>\n"
            f"📉 RSI(40): <b>{rsi_curr:.2f}</b> crossed ↓ WMA(15): <b>{wma_curr:.2f}</b>\n"
            f"🕐 UTC    : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
        )
        send_telegram(msg)

    else:
        log.debug(f"No crossover {symbol} {tf} | RSI {rsi_curr:.2f} | WMA {wma_curr:.2f} | state={current_state}")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main() -> None:
    log.info("=" * 55)
    log.info("  RSI(40)/WMA(15) Crossover Bot  |  Binance Futures")
    log.info(f"  Symbols    : {SYMBOLS}")
    log.info(f"  Timeframes : {TIMEFRAMES}")
    log.info(f"  Poll every : {POLL_INTERVAL}s")
    log.info("=" * 55)

    # Startup ping
    send_telegram(
        "🤖 <b>RSI/WMA Alert Bot Started</b>\n"
        f"Watching: {', '.join(SYMBOLS)}\n"
        f"Timeframes: {', '.join(TIMEFRAMES)}\n"
        f"Signal: RSI(40) ✕ WMA(15)"
    )

    while True:
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                try:
                    check_crossover(symbol, tf)
                except Exception as e:
                    log.error(f"Unhandled error {symbol} {tf}: {e}")
                time.sleep(0.3)   # small delay between API calls

        log.info(f"Cycle done. Sleeping {POLL_INTERVAL}s …")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
