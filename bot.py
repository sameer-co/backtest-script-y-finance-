import backtrader as bt
import pandas as pd
import requests
import time
import collections
import numpy as np
from datetime import datetime, timedelta

# ── Compatibility Patch ───────────────────────────────────────────────────────
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "TOKEN"             : "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg",
    "CHAT_ID"           : "1950462171",

    # ── ACCOUNT ───────────────────────────────────────────────────────────
    "INITIAL_USDC"      : 1000,        # Starting capital in USDC
    "RISK_PERCENT"      : 0.01,        # 1% risk per trade

    # ── SOL DATA ──────────────────────────────────────────────────────────
    "SYMBOL"            : "SOLUSDT",
    "EXCHANGE"          : "Binance",
    "BACKTEST_DAYS"     : 730,         # ~2 years of data

    # ── TIMEFRAMES TO TEST ────────────────────────────────────────────────
    # Binance intervals: "5m", "15m", "1h"
    "TEST_TIMEFRAMES"   : ["5m", "15m", "1h"],

    # ── STRATEGY PARAMS ───────────────────────────────────────────────────
    "ATR_PERIOD"        : 14,
    "ATR_TRAIL_MULT"    : 1.0,         # Trailing stop = 1x ATR after half exit
    "RR_RATIO"          : 2.2,         # Target = SL distance × 2.2

    # ── GENERATION CONFIG ─────────────────────────────────────────────────
    "GEN_START_MA"      : 9,
    "GEN_STEP"          : 2,
    "TOTAL_GENERATIONS" : 10,

    # ── MA TYPES ──────────────────────────────────────────────────────────
    "MA_TYPES"          : ["EMA", "SMA"],

    # ── BINANCE API ───────────────────────────────────────────────────────
    "BINANCE_BASE"      : "https://api.binance.com/api/v3/klines",
    "DELAY_SECONDS"     : 2,
}

# ── Pre-compute generation table ──────────────────────────────────────────────
GENERATIONS = [
    {"gen": g + 1, "ma_p": CONFIG["GEN_START_MA"] + g * CONFIG["GEN_STEP"]}
    for g in range(CONFIG["TOTAL_GENERATIONS"])
]

# ── Interval → milliseconds map ───────────────────────────────────────────────
INTERVAL_MS = {"5m": 5*60*1000, "15m": 15*60*1000, "1h": 60*60*1000}
INTERVAL_LIMIT = 1000   # Binance max candles per request


# ══════════════════════════════════════════════════════════════════════════════
#  BINANCE DATA FETCHER  (no API key needed for public klines)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_binance_klines(symbol, interval, days=730):
    """
    Fetch historical klines from Binance public API.
    Returns a DataFrame with OHLCV columns indexed by datetime.
    Handles pagination automatically for 2 years of data.
    """
    end_ms   = int(datetime.utcnow().timestamp() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000

    all_klines = []
    current_start = start_ms

    print(f"    ⏳ Fetching {symbol} {interval} from Binance...")

    while current_start < end_ms:
        params = {
            "symbol"   : symbol,
            "interval" : interval,
            "startTime": current_start,
            "endTime"  : end_ms,
            "limit"    : INTERVAL_LIMIT,
        }
        try:
            resp = requests.get(CONFIG["BINANCE_BASE"], params=params, timeout=15)
            data = resp.json()
        except Exception as e:
            print(f"    ❌ Fetch error: {e}")
            break

        if not data or not isinstance(data, list):
            break

        all_klines.extend(data)

        # Next page starts after the last candle's close time
        last_close_time = data[-1][6]
        current_start   = last_close_time + 1

        if len(data) < INTERVAL_LIMIT:
            break   # No more pages

        time.sleep(0.3)  # Be polite to Binance API

    if not all_klines:
        return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=[
        "Open_time","Open","High","Low","Close","Volume",
        "Close_time","Quote_vol","Trades","Taker_base","Taker_quote","Ignore"
    ])
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms", utc=True)
    df.set_index("Open_time", inplace=True)
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = df[col].astype(float)
    df = df[["Open","High","Low","Close","Volume"]]
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    print(f"    ✅ {len(df)} candles fetched ({interval})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY
#  Entry  : Close crosses above EMA/SMA
#  SL     : crossover candle low  vs  ATR → use whichever is LARGER
#            i.e. SL distance = max(entry - candle_low, ATR)
#  Target : entry + (SL_distance × 2.2)
#  Exit 1 : sell 50% at target
#  Exit 2 : trail remaining by 1×ATR
#  Exit 3 : hard stop if low ≤ stop_loss
# ══════════════════════════════════════════════════════════════════════════════
class SOLCrossStrategy(bt.Strategy):
    params = (
        ('ma_p'      , 9),
        ('ma_type'   , 'EMA'),
        ('atr_p'     , 14),
        ('rr'        , 2.2),
        ('atr_trail' , 1.0),
    )

    def __init__(self):
        if self.p.ma_type == 'EMA':
            self.ma = bt.indicators.EMA(self.data.close, period=self.p.ma_p)
        else:
            self.ma = bt.indicators.SMA(self.data.close, period=self.p.ma_p)

        self.atr        = bt.indicators.ATR(self.data, period=self.p.atr_p)
        self.crossover  = bt.indicators.CrossOver(self.data.close, self.ma)

        self.stop_loss   = None
        self.target      = None
        self.half_booked = False

        # Tracking counters (per run)
        self.trail_wins  = 0   # exits via trailing stop with profit
        self.target_wins = 0   # exits via target hit (first half)
        self.total_loss  = 0.0 # total $ lost on losing trades

    def next(self):
        if not self.position:
            if self.crossover[0] == 1:
                entry         = self.data.close[0]
                candle_low    = self.data.low[0]   # crossover candle low
                atr_val       = self.atr[0]

                # SL distance = max(entry - candle_low, ATR)
                sl_from_low   = entry - candle_low
                sl_distance   = max(sl_from_low, atr_val)
                stop          = entry - sl_distance
                target        = entry + (sl_distance * self.p.rr)

                if sl_distance > 0:
                    risk_usdc = self.broker.get_value() * CONFIG["RISK_PERCENT"]
                    qty       = risk_usdc / sl_distance
                    cash      = self.broker.get_cash()
                    final_qty = min(qty, cash / entry)
                    if final_qty > 0.0001:
                        self.buy(size=final_qty)
                        self.stop_loss   = stop
                        self.target      = target
                        self.half_booked = False

        elif self.position:
            # ── Target hit → book 50% ────────────────────────────────────
            if not self.half_booked and self.data.high[0] >= self.target:
                self.sell(size=self.position.size / 2)
                self.half_booked = True
                self.stop_loss   = max(self.stop_loss, self.data.open[0])
                self.target_wins += 1

            # ── Trail remaining by 1×ATR ──────────────────────────────────
            if self.half_booked:
                new_stop = self.data.close[0] - (self.atr[0] * self.p.atr_trail)
                if new_stop > self.stop_loss:
                    self.stop_loss = new_stop

            # ── Hard stop hit ─────────────────────────────────────────────
            if self.data.low[0] <= self.stop_loss:
                if self.half_booked:
                    # Second half stopped out after target → still a trail win
                    self.trail_wins += 1
                self.close()

    def notify_trade(self, trade):
        if trade.isclosed and trade.pnl < 0:
            self.total_loss += abs(trade.pnl)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def send_msg(text):
    url = f"https://api.telegram.org/bot{CONFIG['TOKEN']}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": CONFIG["CHAT_ID"], "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        print(f"[Telegram Error] {e}")


def make_gen_stats(gen_num, ma_p, ma_type):
    return {
        "gen"        : gen_num,
        "ma_p"       : ma_p,
        "ma_type"    : ma_type,
        "profit"     : 0.0,
        "trades"     : 0,
        "wins"       : 0,
        "gross_p"    : 0.0,
        "gross_l"    : 0.0,
        "max_dd"     : 0.0,
        "trail_wins" : 0,
        "target_wins": 0,
        "total_loss" : 0.0,
        "best_tf"    : {"name": "", "profit": -999999},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PER-GENERATION REPORT
# ══════════════════════════════════════════════════════════════════════════════
def send_gen_report(gs):
    win_rate      = (gs["wins"] / gs["trades"] * 100) if gs["trades"] > 0 else 0
    profit_factor = (gs["gross_p"] / gs["gross_l"])   if gs["gross_l"] > 0 else 0
    expectancy    = (gs["profit"] / gs["trades"])      if gs["trades"] > 0 else 0
    losses        = gs["trades"] - gs["wins"]

    pf_emoji = "🟢" if profit_factor >= 1.5 else ("🟡" if profit_factor >= 1.0 else "🔴")
    wr_emoji = "🟢" if win_rate >= 50 else "🔴"

    report = (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧬 *GEN {gs['gen']} | SOL/USDT | {gs['ma_type']}({gs['ma_p']})*\n"
        f"📌 *Signal:* Close × {gs['ma_type']} | SL: max(CandleLow, ATR)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *Net P/L:*        ${gs['profit']:,.2f}\n"
        f"{wr_emoji} *Win Rate:*      {win_rate:.2f}%\n"
        f"{pf_emoji} *Profit Factor:* {profit_factor:.2f}\n"
        f"📊 *Total Trades:*   {gs['trades']}\n"
        f"✅ *Wins:*           {gs['wins']}\n"
        f"❌ *Losses:*         {losses}\n"
        f"📉 *Max Drawdown:*   {gs['max_dd']:.2f}%\n"
        f"🎯 *Expectancy:*     ${expectancy:.2f}/trade\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏹 *EXIT BREAKDOWN*\n"
        f"🎯 Target Wins:     {gs['target_wins']} (50% booked at 2.2R)\n"
        f"🔁 Trail Wins:      {gs['trail_wins']} (remainder trailed)\n"
        f"💸 Total Loss:      ${gs['total_loss']:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏱ *Best Timeframe:* {gs['best_tf']['name']}\n"
        f"   Profit: ${gs['best_tf']['profit']:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━"
    )
    print(report)
    send_msg(report)


# ══════════════════════════════════════════════════════════════════════════════
#  PER MA-TYPE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def send_matype_summary(all_stats, ma_type):
    best_gen  = max(all_stats, key=lambda x: x["profit"])
    worst_gen = min(all_stats, key=lambda x: x["profit"])

    table = ""
    for gs in all_stats:
        wr  = (gs["wins"] / gs["trades"] * 100) if gs["trades"] > 0 else 0
        pf  = (gs["gross_p"] / gs["gross_l"])   if gs["gross_l"] > 0 else 0
        exp = (gs["profit"] / gs["trades"])      if gs["trades"] > 0 else 0
        arrow = "⭐" if gs["gen"] == best_gen["gen"] else ("🔴" if gs["gen"] == worst_gen["gen"] else "▪️")
        table += (
            f"{arrow} *G{gs['gen']}* {ma_type}{gs['ma_p']} | "
            f"${gs['profit']:,.0f} | {wr:.1f}% | PF:{pf:.2f} | "
            f"Exp:${exp:.1f} | T:{gs['trades']}\n"
        )

    total_profit   = sum(g["profit"]      for g in all_stats)
    total_trades   = sum(g["trades"]      for g in all_stats)
    total_wins     = sum(g["wins"]        for g in all_stats)
    total_gross_p  = sum(g["gross_p"]     for g in all_stats)
    total_gross_l  = sum(g["gross_l"]     for g in all_stats)
    total_t_wins   = sum(g["target_wins"] for g in all_stats)
    total_tr_wins  = sum(g["trail_wins"]  for g in all_stats)
    total_loss_usd = sum(g["total_loss"]  for g in all_stats)
    overall_wr     = (total_wins / total_trades * 100) if total_trades > 0 else 0
    overall_pf     = (total_gross_p / total_gross_l)   if total_gross_l > 0 else 0
    overall_exp    = (total_profit / total_trades)      if total_trades > 0 else 0

    profits        = [g["profit"] for g in all_stats]
    trend          = "📈 IMPROVING" if profits[-1] > profits[0] else "📉 DECLINING"
    f5_avg         = np.mean(profits[:5])
    l5_avg         = np.mean(profits[5:])

    summary = (
        f"🏛 *{ma_type} — 10 GEN SUMMARY | SOL/USDT*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 *GENERATION TABLE*\n"
        f"*(P/L | WR | PF | Exp | Trades)*\n\n"
        f"{table}"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *AGGREGATED STATS*\n"
        f"💰 Total P/L:     ${total_profit:,.2f}\n"
        f"✅ Win Rate:       {overall_wr:.2f}%\n"
        f"⚖️ Profit Factor:  {overall_pf:.2f}\n"
        f"🎯 Expectancy:     ${overall_exp:.2f}/trade\n"
        f"📊 Total Trades:   {total_trades}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏹 *EXIT BREAKDOWN*\n"
        f"🎯 Target Wins:   {total_t_wins}\n"
        f"🔁 Trail Wins:    {total_tr_wins}\n"
        f"💸 Total Loss:    ${total_loss_usd:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏆 *BEST GEN:*  G{best_gen['gen']} {ma_type}({best_gen['ma_p']}) | ${best_gen['profit']:,.2f}\n"
        f"💀 *WORST GEN:* G{worst_gen['gen']} {ma_type}({worst_gen['ma_p']}) | ${worst_gen['profit']:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 *TREND:* {trend}\n"
        f"   First 5 avg: ${f5_avg:,.0f} → Last 5 avg: ${l5_avg:,.0f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 *BEST PARAM:* {ma_type}({best_gen['ma_p']}) → Gen {best_gen['gen']}\n"
        f"🏁 *{ma_type} Test Complete.*"
    )
    print(summary)
    send_msg(summary)


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER EMA vs SMA REPORT
# ══════════════════════════════════════════════════════════════════════════════
def send_master_report(all_results):
    msg = (
        f"🏆 *MASTER REPORT — EMA vs SMA*\n"
        f"📌 *SOL/USDT | Binance | 2Y Data*\n"
        f"⏱ *TFs: 5m, 15m, 1h | Risk: 1% of $1000*\n"
        f"🧬 *10 Gens | MA 9→27 | Step +2*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    summary_rows = {}
    for ma_type, stats_list in all_results.items():
        best    = max(stats_list, key=lambda x: x["profit"])
        worst   = min(stats_list, key=lambda x: x["profit"])
        total_p = sum(g["profit"]      for g in stats_list)
        total_t = sum(g["trades"]      for g in stats_list)
        total_w = sum(g["wins"]        for g in stats_list)
        gp      = sum(g["gross_p"]     for g in stats_list)
        gl      = sum(g["gross_l"]     for g in stats_list)
        t_wins  = sum(g["target_wins"] for g in stats_list)
        tr_wins = sum(g["trail_wins"]  for g in stats_list)
        t_loss  = sum(g["total_loss"]  for g in stats_list)
        wr      = (total_w / total_t * 100) if total_t > 0 else 0
        pf      = (gp / gl)                 if gl > 0      else 0
        exp     = (total_p / total_t)       if total_t > 0 else 0

        summary_rows[ma_type] = {
            "total_p": total_p, "wr": wr, "pf": pf, "exp": exp,
            "total_t": total_t, "best": best, "worst": worst,
            "t_wins": t_wins, "tr_wins": tr_wins, "t_loss": t_loss
        }

        msg += (
            f"📌 *{ma_type} OVERVIEW*\n"
            f"   💰 Total P/L:    ${total_p:,.2f}\n"
            f"   ✅ Win Rate:     {wr:.2f}%\n"
            f"   ⚖️ Profit Factor:{pf:.2f}\n"
            f"   🎯 Expectancy:   ${exp:.2f}/trade\n"
            f"   📊 Trades:       {total_t}\n"
            f"   🎯 Target Wins:  {t_wins}\n"
            f"   🔁 Trail Wins:   {tr_wins}\n"
            f"   💸 Total Loss:   ${t_loss:,.2f}\n"
            f"   🏆 Best Gen:     G{best['gen']} {ma_type}({best['ma_p']}) ${best['profit']:,.2f}\n"
            f"   💀 Worst Gen:    G{worst['gen']} {ma_type}({worst['ma_p']}) ${worst['profit']:,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    # ── Head-to-head verdict ──────────────────────────────────────────────
    if "EMA" in summary_rows and "SMA" in summary_rows:
        ema = summary_rows["EMA"]
        sma = summary_rows["SMA"]

        pl_winner  = "EMA" if ema["total_p"] >= sma["total_p"] else "SMA"
        wr_winner  = "EMA" if ema["wr"]      >= sma["wr"]      else "SMA"
        pf_winner  = "EMA" if ema["pf"]      >= sma["pf"]      else "SMA"
        exp_winner = "EMA" if ema["exp"]      >= sma["exp"]     else "SMA"
        pl_diff    = abs(ema["total_p"] - sma["total_p"])

        # Score: count how many categories each won
        scores = {"EMA": 0, "SMA": 0}
        for w in [pl_winner, wr_winner, pf_winner, exp_winner]:
            scores[w] += 1
        overall_winner = "EMA" if scores["EMA"] >= scores["SMA"] else "SMA"

        msg += (
            f"⚔️ *HEAD-TO-HEAD VERDICT*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Better P/L:      *{pl_winner}* (by ${pl_diff:,.2f})\n"
            f"✅ Better Win Rate: *{wr_winner}* "
            f"(EMA:{ema['wr']:.1f}% vs SMA:{sma['wr']:.1f}%)\n"
            f"⚖️ Better PF:       *{pf_winner}* "
            f"(EMA:{ema['pf']:.2f} vs SMA:{sma['pf']:.2f})\n"
            f"🎯 Better Exp:      *{exp_winner}* "
            f"(EMA:${ema['exp']:.2f} vs SMA:${sma['exp']:.2f})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🥇 *OVERALL WINNER: {overall_winner}*\n"
            f"   (Won {scores[overall_winner]}/4 categories)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    msg += (
        f"📅 Period: 2Y | Capital: $1000 USDC\n"
        f"🏁 *Full SOL Evolutionary Test Complete.*"
    )
    print(msg)
    send_msg(msg)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 55)
    print("  SOL/USDT Evolutionary Backtester")
    print(f"  Capital: ${CONFIG['INITIAL_USDC']} USDC | Risk: 1%")
    print(f"  Timeframes: {CONFIG['TEST_TIMEFRAMES']}")
    print(f"  MA Types: {CONFIG['MA_TYPES']}")
    print(f"  Generations: {CONFIG['TOTAL_GENERATIONS']} (MA {CONFIG['GEN_START_MA']}→"
          f"{CONFIG['GEN_START_MA'] + (CONFIG['TOTAL_GENERATIONS']-1)*CONFIG['GEN_STEP']})")
    print("=" * 55)

    # ── Pre-fetch all SOL data for each timeframe ─────────────────────────────
    print("\n📥 Fetching SOL/USDT data from Binance...\n")
    sol_data = {}
    for tf in CONFIG["TEST_TIMEFRAMES"]:
        df = fetch_binance_klines(
            symbol   = CONFIG["SYMBOL"],
            interval = tf,
            days     = CONFIG["BACKTEST_DAYS"]
        )
        if not df.empty:
            sol_data[tf] = df
        time.sleep(1)

    if not sol_data:
        print("❌ No data fetched. Check your internet connection.")
        exit()

    # Announce start
    send_msg(
        f"🚀 *SOL/USDT Evolutionary Backtest Starting*\n"
        f"📌 Binance | 2Y Data | $1000 USDC | 1% Risk\n"
        f"⏱ TFs: {', '.join(CONFIG['TEST_TIMEFRAMES'])}\n"
        f"🧬 {CONFIG['TOTAL_GENERATIONS']} Gens | MA {CONFIG['GEN_START_MA']}→"
        f"{CONFIG['GEN_START_MA'] + (CONFIG['TOTAL_GENERATIONS']-1)*CONFIG['GEN_STEP']} | Step +2\n"
        f"🔁 MA Types: EMA + SMA\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Total reports: 23 (20 gen + 2 summary + 1 master)"
    )
    time.sleep(2)

    all_results = {ma_type: [] for ma_type in CONFIG["MA_TYPES"]}

    # ── OUTER LOOP: MA Type ───────────────────────────────────────────────────
    for ma_type in CONFIG["MA_TYPES"]:

        print(f"\n{'#'*55}")
        print(f"  🔁 Starting {ma_type} — 10 Generation Run")
        print(f"{'#'*55}")

        send_msg(
            f"🔁 *Starting {ma_type} Evolutionary Test*\n"
            f"🧬 10 Gens | MA 9→27 | Step +2\n"
            f"📌 Signal: Close × {ma_type} | SL: max(CandleLow, ATR) | TGT: 2.2R"
        )
        time.sleep(2)

        # ── INNER LOOP: Generations ───────────────────────────────────────────
        for gen_info in GENERATIONS:
            gen_num = gen_info["gen"]
            ma_p    = gen_info["ma_p"]

            print(f"\n{'='*50}")
            print(f"🧬 {ma_type} | GEN {gen_num} | MA Period = {ma_p}")
            print(f"{'='*50}")

            gs = make_gen_stats(gen_num, ma_p, ma_type)

            # ── Loop: Timeframes ──────────────────────────────────────────────
            for tf, df in sol_data.items():
                print(f"  📊 SOL/USDT @ {tf} | {ma_type}({ma_p})")

                cerebro = bt.Cerebro()
                cerebro.adddata(bt.feeds.PandasData(dataname=df.copy()), name=f"SOL_{tf}")
                cerebro.addstrategy(
                    SOLCrossStrategy,
                    ma_p      = ma_p,
                    ma_type   = ma_type,
                    atr_p     = CONFIG["ATR_PERIOD"],
                    rr        = CONFIG["RR_RATIO"],
                    atr_trail = CONFIG["ATR_TRAIL_MULT"],
                )
                cerebro.broker.setcash(CONFIG["INITIAL_USDC"])
                cerebro.broker.setcommission(commission=0.001)  # 0.1% Binance fee
                cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tr")
                cerebro.addanalyzer(bt.analyzers.DrawDown,       _name="dd")

                try:
                    results  = cerebro.run()
                    strat    = results[0]
                    res      = strat.analyzers.tr.get_analysis()
                    dd       = strat.analyzers.dd.get_analysis()

                    if 'total' in res and res.total.total > 0:
                        pnl = cerebro.broker.get_value() - CONFIG["INITIAL_USDC"]

                        gs["profit"]      += pnl
                        gs["trades"]      += res.total.total
                        gs["wins"]        += res.won.total           if 'won'  in res else 0
                        gs["gross_p"]     += res.won.pnl.total       if 'won'  in res else 0
                        gs["gross_l"]     += abs(res.lost.pnl.total) if 'lost' in res else 0
                        gs["max_dd"]       = max(gs["max_dd"], dd.max.drawdown)
                        gs["trail_wins"]  += strat.trail_wins
                        gs["target_wins"] += strat.target_wins
                        gs["total_loss"]  += strat.total_loss

                        if pnl > gs["best_tf"]["profit"]:
                            gs["best_tf"] = {"name": f"SOL/USDT ({tf})", "profit": pnl}

                except Exception as e:
                    print(f"    ❌ Error @ {tf}: {e}")

                time.sleep(CONFIG["DELAY_SECONDS"])

            # ── Generation complete → send report ─────────────────────────────
            all_results[ma_type].append(gs)
            send_gen_report(gs)
            print(f"\n✅ {ma_type} Gen {gen_num} report sent.\n")
            time.sleep(2)

        # ── All 10 gens of this MA type done ──────────────────────────────────
        send_matype_summary(all_results[ma_type], ma_type)
        print(f"\n✅ {ma_type} 10-Gen Summary sent.\n")
        time.sleep(3)

    # ── All MA types done → Master Report ────────────────────────────────────
    print("\n🏆 Sending Master EMA vs SMA Report...\n")
    send_master_report(all_results)
    print("🏁 All done.")
