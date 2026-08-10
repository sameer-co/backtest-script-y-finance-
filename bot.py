"""
╔══════════════════════════════════════════════════════════════════╗
║   SOL Renko ATR Bot — Jupiter (Solana) execution, 5m signals     ║
║   Signal source : Binance public candles (SOLUSDT, 5m)           ║
║   Execution     : Jupiter Quote/Swap API on Solana mainnet       ║
╚══════════════════════════════════════════════════════════════════╝

MODES
  DRY_RUN = True   (default, hard-coded safe default)
    - Pulls REAL live quotes from Jupiter so simulated fills reflect
      real routing/slippage, but NEVER builds, signs, or sends a
      transaction. Tracks a virtual paper balance only.
    - No private key required in this mode.

  DRY_RUN = False  (live trading -- real money, irreversible)
    - Signs and sends real swap transactions using a keypair loaded
      from the JUP_WALLET_PRIVATE_KEY environment variable (base58
      string). NEVER hardcode a real private key into this file.
    - Use a burner/trading-only wallet funded with a SMALL amount.
      You always need a little SOL left over for network fees even
      when your position is fully in USDC.
    - Start with a tiny position_size_pct and watch the first few
      live trades closely before trusting it unattended.

REQUIREMENTS
  pip install requests numpy solders base58 --break-system-packages

  Environment variables (live mode only):
    JUP_WALLET_PRIVATE_KEY   base58-encoded Solana secret key

IMPORTANT LIMITS I CANNOT VERIFY FROM HERE
  I cannot reach api.jup.ag, quote-api.jup.ag, or any Solana RPC from
  my sandbox (network is allowlisted to a small set of dev domains),
  so the Jupiter/Solana calls below are written carefully but UNTESTED
  against the live network. Run this in DRY_RUN mode first and watch
  the console + Telegram output before ever flipping DRY_RUN to False.

  Also note: Binance's SOLUSDT price (used for signals) and Jupiter's
  on-chain SOL/USDC price (used for fills) are two different venues.
  They usually track closely but can diverge briefly during volatility
  -- that basis risk is real and not something this script hedges.

NOT FINANCIAL ADVICE. You are responsible for your own funds and risk.
"""

import os
import csv
import time
import base64
import requests
import numpy as np
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────
#  MODE SWITCH -- leave this True until you have fully validated
#  behaviour in dry-run and are ready to risk real funds.
# ─────────────────────────────────────────────────────────────
DRY_RUN = True

# ─────────────────────────────────────────────────────────────
#  TELEGRAM CONFIG -- rotate this token, it has been pasted in
#  chat previously and should be treated as compromised.
# ─────────────────────────────────────────────────────────────
TG_TOKEN   = "8661081060:AAGtNViZMS6FSl_7vQeMz1TcCnzrFddu7z4"
TG_CHAT_ID = "1950462171"
TG_URL     = f"https://api.telegram.org/bot{TG_TOKEN}"

# ─────────────────────────────────────────────────────────────
#  SIGNAL SOURCE (Binance public candles, same strategy logic
#  as the forward tester / backtester)
# ─────────────────────────────────────────────────────────────
BINANCE_SYMBOL   = "SOLUSDT"
BINANCE_INTERVAL = "5m"
BINANCE_KLINES_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]
BINANCE_TICKER_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/ticker/price",
    "https://api.binance.com/api/v3/ticker/price",
]
LOOKBACK_CANDLES = 200
POLL_SECONDS     = 5 * 60

SETTINGS = {
    "atr_period"      : 14,
    "renko_mult"      : 1.0,
    "sl_mult"         : 1.5,
    "tp_mult"         : 3.0,
    "min_sell_bricks" : 2,
}

# ─────────────────────────────────────────────────────────────
#  JUPITER / SOLANA EXECUTION
# ─────────────────────────────────────────────────────────────
JUP_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUP_SWAP_URL  = "https://quote-api.jup.ag/v6/swap"
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"  # swap for your own RPC for reliability

SOL_MINT  = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SOL_DECIMALS  = 9
USDC_DECIMALS = 6

SLIPPAGE_BPS       = 50     # 0.50% max slippage tolerance per swap on Jupiter
POSITION_SIZE_PCT  = 0.20   # fraction of available USDC risked per trade
START_PAPER_USDC   = 1000.0 # virtual starting balance for dry-run mode

TRADES_CSV = "jupiter_renko_trades.csv"
DAILY_REPORT_SECONDS = 24 * 60 * 60   # send a daily CSV summary every 24h


# ─────────────────────────────────────────────────────────────
#  TELEGRAM HELPERS
# ─────────────────────────────────────────────────────────────
def send_tg(message: str, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            r = requests.post(f"{TG_URL}/sendMessage", json={
                "chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"
            }, timeout=10)
            if r.status_code == 200:
                return True
            print(f"[TG] HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[TG] error attempt {attempt+1}: {e}")
        time.sleep(2)
    return False


def send_tg_document(path: str, caption: str = "", retries: int = 3) -> bool:
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


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt(p: float) -> str:
    return f"{p:.4f}"


# ─────────────────────────────────────────────────────────────
#  BINANCE DATA (signal source only -- no execution here)
# ─────────────────────────────────────────────────────────────
def _get_json(endpoints, params, retries=5):
    last_err = None
    for base_url in endpoints:
        for attempt in range(retries):
            try:
                r = requests.get(base_url, params=params, timeout=15)
                if r.status_code == 451:
                    last_err = RuntimeError(f"451 from {base_url}")
                    break
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(2)
    raise RuntimeError(f"All endpoints failed: {last_err}")


def fetch_candles(symbol: str, interval: str, limit: int):
    params = {"symbol": symbol, "interval": interval, "limit": limit + 1}
    raw = _get_json(BINANCE_KLINES_ENDPOINTS, params)[:-1]  # drop live candle
    closes = np.array([float(c[4]) for c in raw])
    highs  = np.array([float(c[2]) for c in raw])
    lows   = np.array([float(c[3]) for c in raw])
    return highs, lows, closes


def fetch_ticker_price(symbol: str) -> float:
    data = _get_json(BINANCE_TICKER_ENDPOINTS, {"symbol": symbol})
    return float(data["price"])


def calc_atr(highs, lows, closes, period: int) -> np.ndarray:
    n, tr, atr, s = len(closes), np.zeros(len(closes)), np.zeros(len(closes)), 0.0
    for i in range(1, n):
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        if i < period:
            s += tr[i]
        elif i == period:
            s += tr[i]
            atr[i] = s / period
        else:
            atr[i] = (atr[i-1]*(period-1) + tr[i]) / period
    return atr


def build_renko(closes, atr_arr, mult: float):
    bricks, ref, ref_atr = [], None, None
    for i in range(len(closes)):
        a = atr_arr[i]
        if a == 0:
            continue
        if ref is None:
            ref, ref_atr = closes[i], a
            continue
        price = closes[i]
        brick_sz = ref_atr * mult
        while price >= ref + brick_sz:
            bricks.append({"dir": 1, "close": ref+brick_sz, "idx": i, "atr": ref_atr})
            ref += brick_sz; ref_atr = a; brick_sz = ref_atr * mult
        while price <= ref - brick_sz:
            bricks.append({"dir": -1, "close": ref-brick_sz, "idx": i, "atr": ref_atr})
            ref -= brick_sz; ref_atr = a; brick_sz = ref_atr * mult
    return bricks


def detect_signal(bricks, min_sell_bricks, sl_mult, tp_mult,
                   last_brick_count, last_entry_price=0.0, atr_gap_mult=1.0):
    n = len(bricks)
    if n == last_brick_count or n < 3:
        return None, n
    sell_run, signal = 0, None
    for i, b in enumerate(bricks):
        if b["dir"] == -1:
            sell_run += 1
        else:
            if sell_run >= min_sell_bricks and i >= last_brick_count:
                entry, atr = b["close"], b["atr"]
                if last_entry_price > 0 and abs(entry - last_entry_price) < atr_gap_mult * atr:
                    sell_run = 0
                    continue
                sl = entry - sl_mult * atr
                tp = entry + tp_mult * sl_mult * atr
                signal = {"entry": entry, "sl": sl, "tp": tp, "atr": atr, "sell_run": sell_run}
            sell_run = 0
    return signal, n


# ─────────────────────────────────────────────────────────────
#  JUPITER QUOTE / SWAP
# ─────────────────────────────────────────────────────────────
def get_jupiter_quote(input_mint: str, output_mint: str, amount_base_units: int,
                       slippage_bps: int = SLIPPAGE_BPS):
    """
    amount_base_units: integer amount in the input token's smallest unit
    (lamports for SOL, 6-decimal units for USDC).
    Returns Jupiter's quote response dict (includes outAmount, price impact, route).
    """
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": int(amount_base_units),
        "slippageBps": slippage_bps,
        "swapMode": "ExactIn",
    }
    r = requests.get(JUP_QUOTE_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def execute_jupiter_swap(quote_response: dict, keypair, rpc_url: str = SOLANA_RPC_URL):
    """
    LIVE ONLY. Builds, signs, and sends the swap transaction returned by
    Jupiter for a given quote. Requires `solders`.
    Returns the transaction signature (string) once submitted.
    """
    from solders.transaction import VersionedTransaction  # noqa: local import, live-mode only

    swap_resp = requests.post(JUP_SWAP_URL, json={
        "quoteResponse": quote_response,
        "userPublicKey": str(keypair.pubkey()),
        "wrapAndUnwrapSol": True,
        "dynamicComputeUnitLimit": True,
        "prioritizationFeeLamports": "auto",
    }, timeout=20)
    swap_resp.raise_for_status()
    swap_tx_b64 = swap_resp.json()["swapTransaction"]

    raw_tx = base64.b64decode(swap_tx_b64)
    unsigned_tx = VersionedTransaction.from_bytes(raw_tx)
    signed_tx = VersionedTransaction(unsigned_tx.message, [keypair])
    signed_b64 = base64.b64encode(bytes(signed_tx)).decode("utf-8")

    rpc_resp = requests.post(rpc_url, json={
        "jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
        "params": [signed_b64, {"skipPreflight": False, "encoding": "base64",
                                 "maxRetries": 3}],
    }, timeout=20)
    rpc_resp.raise_for_status()
    result = rpc_resp.json()
    if "error" in result:
        raise RuntimeError(f"sendTransaction failed: {result['error']}")
    return result["result"]  # tx signature


def load_live_keypair():
    """LIVE ONLY. Loads the trading wallet from JUP_WALLET_PRIVATE_KEY."""
    from solders.keypair import Keypair  # noqa: local import, live-mode only
    import base58

    key_b58 = os.environ.get("JUP_WALLET_PRIVATE_KEY")
    if not key_b58:
        raise RuntimeError(
            "JUP_WALLET_PRIVATE_KEY environment variable not set. "
            "Live mode requires a base58-encoded secret key for your "
            "burner trading wallet."
        )
    secret = base58.b58decode(key_b58)
    return Keypair.from_bytes(secret)


# ─────────────────────────────────────────────────────────────
#  TRADE LOG
# ─────────────────────────────────────────────────────────────
def log_trade(row: dict):
    file_exists = os.path.isfile(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# ─────────────────────────────────────────────────────────────
#  DAILY REPORT (reads TRADES_CSV, summarizes last 24h + all-time)
# ─────────────────────────────────────────────────────────────
def read_trades_csv(path: str):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_daily_report_text(path: str, current_balance_usdc: float = None) -> str:
    rows = read_trades_csv(path)
    if not rows:
        return (
            f"🗒️ <b>Daily Report — SOL/USDC bot</b>\n"
            f"No trades logged yet in <code>{path}</code>.\n"
            f"⏰ {now_utc()}"
        )

    now = datetime.now(timezone.utc)
    last_24h = []
    for row in rows:
        try:
            exit_dt = datetime.strptime(row["exit_time"], "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if (now - exit_dt).total_seconds() <= DAILY_REPORT_SECONDS:
            last_24h.append(row)

    def summarize(subset):
        n = len(subset)
        wins   = [r for r in subset if r.get("reason") == "TP"]
        losses = [r for r in subset if r.get("reason") == "SL"]
        pnl_usdc = sum(float(r["pnl_usdc"]) for r in subset) if subset else 0.0
        winrate = (len(wins) / n * 100) if n else 0.0
        return n, len(wins), len(losses), winrate, pnl_usdc

    n24, w24, l24, wr24, pnl24 = summarize(last_24h)
    nall, wall, lall, wrall, pnlall = summarize(rows)

    mode = rows[-1].get("mode", "?") if rows else "?"
    lines = [
        f"🗒️ <b>Daily Report — SOL/USDC bot</b> ({mode})",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"<b>Last 24h</b>",
        f"Trades: {n24}  |  Win/Loss: {w24}/{l24}  |  Win rate: {wr24:.1f}%",
        f"P&amp;L: <b>{pnl24:+.2f} USDC</b>",
        f"",
        f"<b>All-time ({nall} trades)</b>",
        f"Win/Loss: {wall}/{lall}  |  Win rate: {wrall:.1f}%",
        f"Total P&amp;L: <b>{pnlall:+.2f} USDC</b>",
    ]
    if current_balance_usdc is not None:
        lines.append(f"")
        lines.append(f"Current balance: <code>{current_balance_usdc:.2f} USDC</code>")
    lines.append(f"⏰ {now_utc()}")
    return "\n".join(lines)


def send_daily_report(csv_path: str, current_balance_usdc: float = None):
    text = build_daily_report_text(csv_path, current_balance_usdc)
    send_tg(text)
    if os.path.isfile(csv_path):
        send_tg_document(csv_path, caption=f"Full trade log as of {now_utc()}")
    print(f"[BOT] daily report sent")


# ─────────────────────────────────────────────────────────────
#  BOT
# ─────────────────────────────────────────────────────────────
class JupiterRenkoBot:
    def __init__(self):
        self.last_n = 0
        self.last_entry_price = 0.0
        self.position = None   # dict when in a trade, else None
        self.last_daily_report = time.time()

        if DRY_RUN:
            self.usdc_balance = START_PAPER_USDC
            self.sol_balance  = 0.0
            self.keypair = None
        else:
            self.keypair = load_live_keypair()
            self.usdc_balance = None  # fetched live before each trade
            self.sol_balance  = None

        mode_str = "DRY RUN (paper, real quotes)" if DRY_RUN else "LIVE (real funds!)"
        send_tg(
            f"🤖 <b>Jupiter Renko Bot Started</b>\n"
            f"Mode: <b>{mode_str}</b>\n"
            f"Pair: SOL/USDC | Signal TF: {BINANCE_INTERVAL}\n"
            f"Position size: {POSITION_SIZE_PCT*100:.0f}% of balance per trade\n"
            f"Slippage tolerance: {SLIPPAGE_BPS/100:.2f}%\n"
            f"⏰ {now_utc()}"
        )
        print(f"[BOT] started, mode={mode_str}")

    # ── main loop ────────────────────────────────────────────
    def run(self):
        while True:
            try:
                self._tick()
            except Exception as e:
                msg = f"⚠️ Bot error: {e}"
                print(msg)
                send_tg(msg)

            if time.time() - self.last_daily_report >= DAILY_REPORT_SECONDS:
                try:
                    bal = self._current_balance_estimate()
                    send_daily_report(TRADES_CSV, current_balance_usdc=bal)
                except Exception as e:
                    print(f"[BOT] daily report error: {e}")
                self.last_daily_report = time.time()

            time.sleep(POLL_SECONDS)

    def _current_balance_estimate(self):
        """Best-effort balance snapshot for the daily report footer."""
        if DRY_RUN:
            try:
                price = fetch_ticker_price(BINANCE_SYMBOL)
            except Exception:
                price = 0.0
            return self.usdc_balance + self.sol_balance * price
        return None  # wire up real balance fetch once _fetch_live_usdc_balance is implemented

    # ── one poll cycle ──────────────────────────────────────
    def _tick(self):
        highs, lows, closes = fetch_candles(BINANCE_SYMBOL, BINANCE_INTERVAL, LOOKBACK_CANDLES)
        atr_arr = calc_atr(highs, lows, closes, SETTINGS["atr_period"])
        bricks  = build_renko(closes, atr_arr, SETTINGS["renko_mult"])

        if self.position is not None:
            self._check_exit()
            return

        signal, new_n = detect_signal(
            bricks, SETTINGS["min_sell_bricks"], SETTINGS["sl_mult"], SETTINGS["tp_mult"],
            self.last_n, last_entry_price=self.last_entry_price,
        )
        self.last_n = new_n

        if signal:
            self._enter_trade(signal)
        else:
            print(f"[BOT] {now_utc()} no signal | bricks={new_n} | close={closes[-1]:.4f}")

    # ── entry ────────────────────────────────────────────────
    def _enter_trade(self, signal: dict):
        usdc_avail = self.usdc_balance if DRY_RUN else self._fetch_live_usdc_balance()
        usdc_to_spend = usdc_avail * POSITION_SIZE_PCT
        if usdc_to_spend <= 1.0:
            print("[BOT] position size too small, skipping entry")
            return

        amount_units = int(usdc_to_spend * (10 ** USDC_DECIMALS))
        quote = get_jupiter_quote(USDC_MINT, SOL_MINT, amount_units)
        sol_received = int(quote["outAmount"]) / (10 ** SOL_DECIMALS)
        fill_price = usdc_to_spend / sol_received  # effective USDC per SOL, incl. real slippage

        if DRY_RUN:
            self.usdc_balance -= usdc_to_spend
            self.sol_balance  += sol_received
            tx_sig = "DRY_RUN"
        else:
            tx_sig = execute_jupiter_swap(quote, self.keypair)

        self.position = {
            "entry_time": now_utc(),
            "entry_signal_price": signal["entry"],   # Binance-derived Renko level
            "entry_fill_price": fill_price,           # actual Jupiter execution price
            "sl": signal["sl"],
            "tp": signal["tp"],
            "atr": signal["atr"],
            "usdc_spent": usdc_to_spend,
            "sol_bought": sol_received,
            "tx_sig": tx_sig,
        }
        self.last_entry_price = signal["entry"]

        send_tg(
            f"🟢 <b>BUY — SOL/USDC</b> ({'DRY RUN' if DRY_RUN else 'LIVE'})\n"
            f"Signal price : <code>{fmt(signal['entry'])}</code>\n"
            f"Fill price   : <code>{fmt(fill_price)}</code>\n"
            f"SL / TP      : <code>{fmt(signal['sl'])}</code> / <code>{fmt(signal['tp'])}</code>\n"
            f"USDC spent   : <code>{usdc_to_spend:.2f}</code>\n"
            f"SOL bought   : <code>{sol_received:.4f}</code>\n"
            f"Tx           : <code>{tx_sig}</code>\n"
            f"⏰ {now_utc()}"
        )
        print(f"[BOT] entered trade @ {fill_price:.4f}")

    # ── exit check ───────────────────────────────────────────
    def _check_exit(self):
        p = self.position
        live_price = fetch_ticker_price(BINANCE_SYMBOL)  # cheap monitoring, no Jupiter call yet

        if live_price >= p["tp"]:
            self._close_trade("TP", live_price)
        elif live_price <= p["sl"]:
            self._close_trade("SL", live_price)
        else:
            print(f"[BOT] {now_utc()} open position | price={live_price:.4f} "
                  f"entry={p['entry_fill_price']:.4f}")

    def _close_trade(self, reason: str, trigger_price: float):
        p = self.position
        amount_units = int(p["sol_bought"] * (10 ** SOL_DECIMALS))
        quote = get_jupiter_quote(SOL_MINT, USDC_MINT, amount_units)
        usdc_received = int(quote["outAmount"]) / (10 ** USDC_DECIMALS)
        exit_fill_price = usdc_received / p["sol_bought"]

        if DRY_RUN:
            self.sol_balance -= p["sol_bought"]
            self.usdc_balance += usdc_received
            tx_sig = "DRY_RUN"
        else:
            tx_sig = execute_jupiter_swap(quote, self.keypair)

        pnl_usdc = usdc_received - p["usdc_spent"]
        pnl_pct  = pnl_usdc / p["usdc_spent"] * 100

        log_trade({
            "entry_time": p["entry_time"], "exit_time": now_utc(),
            "reason": reason,
            "entry_fill_price": round(p["entry_fill_price"], 4),
            "exit_fill_price": round(exit_fill_price, 4),
            "usdc_spent": round(p["usdc_spent"], 2),
            "usdc_received": round(usdc_received, 2),
            "pnl_usdc": round(pnl_usdc, 2),
            "pnl_pct": round(pnl_pct, 3),
            "mode": "DRY_RUN" if DRY_RUN else "LIVE",
            "entry_tx": p["tx_sig"], "exit_tx": tx_sig,
        })

        icon = "✅" if reason == "TP" else "❌"
        send_tg(
            f"{icon} <b>{reason} HIT — SOL/USDC</b> ({'DRY RUN' if DRY_RUN else 'LIVE'})\n"
            f"Entry fill : <code>{fmt(p['entry_fill_price'])}</code>\n"
            f"Exit fill  : <code>{fmt(exit_fill_price)}</code>\n"
            f"P&amp;L        : <b>{pnl_usdc:+.2f} USDC ({pnl_pct:+.2f}%)</b>\n"
            f"Tx         : <code>{tx_sig}</code>\n"
            f"⏰ {now_utc()}"
        )
        print(f"[BOT] closed trade ({reason}) pnl={pnl_usdc:+.2f} USDC ({pnl_pct:+.2f}%)")
        self.position = None

    # ── live balance helper (live mode only) ─────────────────
    def _fetch_live_usdc_balance(self) -> float:
        # Minimal RPC call for the wallet's USDC associated token account balance.
        # For production use, resolve the correct ATA and call getTokenAccountBalance.
        raise NotImplementedError(
            "Wire up getTokenAccountBalance for your wallet's USDC ATA here "
            "before running live. Left unimplemented so live mode can't run "
            "unattended without you deliberately filling this in."
        )


def main():
    print(__doc__)
    if not DRY_RUN:
        print("!!! LIVE MODE -- real funds will be traded !!!")
    bot = JupiterRenkoBot()
    bot.run()


if __name__ == "__main__":
    main()
