import backtrader as bt
import pandas as pd
import requests
import time
import collections
import numpy as np
from datetime import datetime

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
    "INITIAL_USDC"      : 1000,
    "RISK_PERCENT"      : 0.01,        # 1% risk per trade

    # ── SOL DATA ──────────────────────────────────────────────────────────
    "SYMBOL"            : "SOLUSDT",
    "BACKTEST_DAYS"     : 730,         # 2 years

    # ── TIMEFRAMES ────────────────────────────────────────────────────────
    "TEST_TIMEFRAMES"   : ["5m", "15m", "1h"],

    # ── STRATEGY ──────────────────────────────────────────────────────────
    "ATR_PERIOD"        : 14,
    "ATR_SL_MULT"       : 2.0,         # SL = entry - (2 × ATR)
    "RR_RATIO"          : 2.2,         # Target = SL distance × 2.2
    "ATR_TRAIL_MULT"    : 1.0,         # Trail = 1 × ATR after half exit

    # ── GENERATION CONFIG ─────────────────────────────────────────────────
    # Gen 1 : RSI=14, WMA=15
    # Gen 2 : RSI=16, WMA=15  (only RSI +2 each gen, WMA fixed)
    # ...
    # Gen 10: RSI=32, WMA=15
    "GEN_START_RSI"     : 14,
    "WMA_FIXED"         : 15,          # WMA stays the same all 10 gens
    "GEN_STEP_RSI"      : 2,
    "TOTAL_GENERATIONS" : 10,

    # ── BINANCE ───────────────────────────────────────────────────────────
    "BINANCE_BASE"      : "https://api.binance.com/api/v3/klines",
    "DELAY_SECONDS"     : 2,
}

# ── Generation table ──────────────────────────────────────────────────────────
# RSI steps +2 each gen, WMA stays fixed at 15
GENERATIONS = [
    {
        "gen"  : g + 1,
        "rsi_p": CONFIG["GEN_START_RSI"] + g * CONFIG["GEN_STEP_RSI"],
        "wma_p": CONFIG["WMA_FIXED"],
    }
    for g in range(CONFIG["TOTAL_GENERATIONS"])
]


# ══════════════════════════════════════════════════════════════════════════════
#  BINANCE DATA FETCHER
# ══════════════════════════════════════════════════════════════════════════════
def fetch_binance_klines(symbol, interval, days=730):
    end_ms        = int(datetime.utcnow().timestamp() * 1000)
    start_ms      = end_ms - days * 24 * 60 * 60 * 1000
    all_klines    = []
    current_start = start_ms

    print(f"    ⏳ Fetching {symbol} {interval} ...")

    while current_start < end_ms:
        params = {
            "symbol"   : symbol,
            "interval" : interval,
            "startTime": current_start,
            "endTime"  : end_ms,
            "limit"    : 1000,
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
        last_close_time = data[-1][6]
        current_start   = last_close_time + 1

        if len(data) < 1000:
            break
        time.sleep(0.3)

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
#  Entry  : RSI crosses ABOVE WMA of RSI (long only)
#  SL     : entry − (2 × ATR)
#  Target : entry + (SL distance × 2.2)
#  Exit 1 : sell 50% at target  → target_win +1
#  Exit 2 : trail remaining by 1×ATR → trail_win +1 when closed
#  Exit 3 : hard stop if low ≤ stop_loss
#  Rule   : only 1 open trade at a time (enforced by position check)
# ══════════════════════════════════════════════════════════════════════════════
class RSIWMACrossStrategy(bt.Strategy):
    params = (
        ('rsi_p'     , 14),
        ('wma_p'     , 15),
        ('atr_p'     , 14),
        ('atr_sl'    , 2.0),
        ('rr'        , 2.2),
        ('atr_trail' , 1.0),
    )

    def __init__(self):
        self.rsi     = bt.indicators.RSI(self.data.close, period=self.p.rsi_p)
        self.wma_rsi = bt.indicators.WMA(self.rsi,        period=self.p.wma_p)
        self.atr     = bt.indicators.ATR(self.data,       period=self.p.atr_p)

        # RSI crossing above WMA → entry signal
        self.cross   = bt.indicators.CrossOver(self.rsi, self.wma_rsi)

        self.stop_loss  = None
        self.target     = None

        # Per-run exit counters
        self.target_wins = 0
        self.total_loss  = 0.0

    def next(self):
        # ── NO POSITION → look for entry ─────────────────────────────────
        if not self.position:
            if self.cross[0] == 1:                      # RSI crossed above WMA
                entry       = self.data.close[0]
                sl_distance = self.p.atr_sl * self.atr[0]   # 2 × ATR
                stop        = entry - sl_distance
                target      = entry + (sl_distance * self.p.rr)

                if sl_distance > 0:
                    risk_usdc = self.broker.get_value() * CONFIG["RISK_PERCENT"]
                    qty       = risk_usdc / sl_distance
                    final_qty = min(qty, self.broker.get_cash() / entry)

                    if final_qty > 0.0001:
                        self.buy(size=final_qty)
                        self.stop_loss = stop
                        self.target    = target

        # ── IN POSITION → manage trade ────────────────────────────────────
        elif self.position:

            # Target hit → exit 100% immediately, no trailing
            if self.data.high[0] >= self.target:
                self.close()
                self.target_wins += 1

            # Hard stop hit
            elif self.data.low[0] <= self.stop_loss:
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


def make_gen_stats(gen_num, rsi_p, wma_p):
    return {
        "gen"        : gen_num,
        "rsi_p"      : rsi_p,
        "wma_p"      : wma_p,
        "profit"     : 0.0,
        "trades"     : 0,
        "wins"       : 0,
        "gross_p"    : 0.0,
        "gross_l"    : 0.0,
        "max_dd"     : 0.0,
        "target_wins": 0,
        "total_loss" : 0.0,
        "best_tf"    : {"name": "", "profit": -999999},
        # per-timeframe breakdown
        "tf_stats"   : {tf: {"profit": 0.0, "trades": 0, "wins": 0}
                        for tf in CONFIG["TEST_TIMEFRAMES"]},
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

    # Timeframe breakdown
    tf_lines = ""
    for tf, td in gs["tf_stats"].items():
        tf_wr  = (td["wins"] / td["trades"] * 100) if td["trades"] > 0 else 0
        tf_lines += (
            f"  ⏱ *{tf}:* ${td['profit']:,.2f} | "
            f"T:{td['trades']} | WR:{tf_wr:.1f}%\n"
        )

    report = (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧬 *GENERATION {gs['gen']} REPORT*\n"
        f"🔧 *RSI({gs['rsi_p']}) × WMA({gs['wma_p']}) | SOL/USDT*\n"
        f"📌 SL: 2×ATR | TGT: 2.2R | Trail: 1×ATR\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *Net P/L:*        ${gs['profit']:,.2f}\n"
        f"{wr_emoji} *Win Rate:*      {win_rate:.2f}%\n"
        f"{pf_emoji} *Profit Factor:* {profit_factor:.2f}\n"
        f"🎯 *Expectancy:*     ${expectancy:.2f}/trade\n"
        f"📊 *Total Trades:*   {gs['trades']}\n"
        f"✅ *Wins:*           {gs['wins']}\n"
        f"❌ *Losses:*         {losses}\n"
        f"📉 *Max Drawdown:*   {gs['max_dd']:.2f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏹 *EXIT BREAKDOWN*\n"
        f"🎯 Target Wins (2.2R full exit): {gs['target_wins']}\n"
        f"🛑 Stop Loss Hits:               {gs['trades'] - gs['wins']}\n"
        f"💸 Total Loss Amount:            ${gs['total_loss']:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏱ *TIMEFRAME BREAKDOWN*\n"
        f"{tf_lines}"
        f"🌟 *Best TF:* {gs['best_tf']['name']} → ${gs['best_tf']['profit']:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━"
    )
    print(report)
    send_msg(report)


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER SUMMARY REPORT  — all 10 generations compared
# ══════════════════════════════════════════════════════════════════════════════
def send_master_report(all_stats):
    best_gen  = max(all_stats, key=lambda x: x["profit"])
    worst_gen = min(all_stats, key=lambda x: x["profit"])

    # Generation comparison table
    table = ""
    for gs in all_stats:
        wr  = (gs["wins"] / gs["trades"] * 100) if gs["trades"] > 0 else 0
        pf  = (gs["gross_p"] / gs["gross_l"])   if gs["gross_l"] > 0 else 0
        exp = (gs["profit"] / gs["trades"])      if gs["trades"] > 0 else 0
        arrow = "⭐" if gs["gen"] == best_gen["gen"] else (
                "🔴" if gs["gen"] == worst_gen["gen"] else "▪️")
        table += (
            f"{arrow} *G{gs['gen']}* RSI{gs['rsi_p']}/WMA{gs['wma_p']} | "
            f"${gs['profit']:,.0f} | {wr:.1f}% | "
            f"PF:{pf:.2f} | Exp:${exp:.1f} | T:{gs['trades']}\n"
        )

    # Aggregated totals
    total_profit   = sum(g["profit"]      for g in all_stats)
    total_trades   = sum(g["trades"]      for g in all_stats)
    total_wins     = sum(g["wins"]        for g in all_stats)
    total_gp       = sum(g["gross_p"]     for g in all_stats)
    total_gl       = sum(g["gross_l"]     for g in all_stats)
    total_t_wins   = sum(g["target_wins"] for g in all_stats)
    total_loss_usd = sum(g["total_loss"]  for g in all_stats)
    overall_wr     = (total_wins / total_trades * 100) if total_trades > 0 else 0
    overall_pf     = (total_gp / total_gl)             if total_gl > 0    else 0
    overall_exp    = (total_profit / total_trades)      if total_trades > 0 else 0

    # Trend: first 5 gens avg vs last 5 gens avg
    profits   = [g["profit"] for g in all_stats]
    trend     = "📈 IMPROVING" if profits[-1] > profits[0] else "📉 DECLINING"
    f5_avg    = np.mean(profits[:5])
    l5_avg    = np.mean(profits[5:])

    # Per-timeframe totals across all generations
    tf_summary = ""
    for tf in CONFIG["TEST_TIMEFRAMES"]:
        tf_p = sum(g["tf_stats"][tf]["profit"] for g in all_stats)
        tf_t = sum(g["tf_stats"][tf]["trades"] for g in all_stats)
        tf_w = sum(g["tf_stats"][tf]["wins"]   for g in all_stats)
        tf_wr = (tf_w / tf_t * 100) if tf_t > 0 else 0
        tf_summary += (
            f"  ⏱ *{tf}:* ${tf_p:,.2f} | T:{tf_t} | WR:{tf_wr:.1f}%\n"
        )

    master = (
        f"🏛 *MASTER EVOLUTIONARY REPORT*\n"
        f"📌 *SOL/USDT | Binance | 2Y | $1000 USDC*\n"
        f"🔧 *RSI({CONFIG['GEN_START_RSI']}→{CONFIG['GEN_START_RSI']+(CONFIG['TOTAL_GENERATIONS']-1)*CONFIG['GEN_STEP_RSI']}) "
        f"× WMA({CONFIG['WMA_FIXED']}) Fixed | Step +2 RSI*\n"
        f"⏱ *TFs: 5m, 15m, 1h | Risk: 1%/trade*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 *GENERATION COMPARISON TABLE*\n"
        f"*(P/L | WR | PF | Exp | Trades)*\n\n"
        f"{table}"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *OVERALL AGGREGATED STATS*\n"
        f"💰 Total P/L:     ${total_profit:,.2f}\n"
        f"✅ Win Rate:       {overall_wr:.2f}%\n"
        f"⚖️ Profit Factor:  {overall_pf:.2f}\n"
        f"🎯 Expectancy:     ${overall_exp:.2f}/trade\n"
        f"📊 Total Trades:   {total_trades}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏹 *EXIT BREAKDOWN (ALL GENS)*\n"
        f"🎯 Target Wins:   {total_t_wins}\n"
        f"🛑 Stop Loss Hits:{total_trades - total_wins}\n"
        f"💸 Total Loss:    ${total_loss_usd:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏱ *TIMEFRAME TOTALS (ALL GENS)*\n"
        f"{tf_summary}"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏆 *BEST GEN:*\n"
        f"   Gen {best_gen['gen']} → RSI({best_gen['rsi_p']}) × WMA({best_gen['wma_p']})\n"
        f"   P/L: ${best_gen['profit']:,.2f}\n\n"
        f"💀 *WORST GEN:*\n"
        f"   Gen {worst_gen['gen']} → RSI({worst_gen['rsi_p']}) × WMA({worst_gen['wma_p']})\n"
        f"   P/L: ${worst_gen['profit']:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 *PARAMETER TREND:* {trend}\n"
        f"   First 5 avg: ${f5_avg:,.0f} → Last 5 avg: ${l5_avg:,.0f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 *OPTIMAL PARAMS:*\n"
        f"   RSI({best_gen['rsi_p']}) × WMA({best_gen['wma_p']}) → Gen {best_gen['gen']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏁 *Evolutionary Test Complete.*"
    )
    print(master)
    send_msg(master)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 55)
    print("  SOL/USDT RSI×WMA Evolutionary Backtester")
    print(f"  Capital: ${CONFIG['INITIAL_USDC']} USDC | Risk: 1%")
    print(f"  Timeframes: {CONFIG['TEST_TIMEFRAMES']}")
    print(f"  RSI: {CONFIG['GEN_START_RSI']}→"
          f"{CONFIG['GEN_START_RSI']+(CONFIG['TOTAL_GENERATIONS']-1)*CONFIG['GEN_STEP_RSI']}"
          f" | WMA: {CONFIG['WMA_FIXED']} (fixed)")
    print(f"  Generations: {CONFIG['TOTAL_GENERATIONS']}")
    print("=" * 55)

    # ── Print generation table to console ────────────────────────────────────
    print("\n📋 Generation Parameter Table:")
    for g in GENERATIONS:
        print(f"  Gen {g['gen']:2d} → RSI({g['rsi_p']:2d}) × WMA({g['wma_p']})")

    # ── Fetch SOL/USDT data for each timeframe (once) ────────────────────────
    print("\n📥 Fetching SOL/USDT data from Binance...\n")
    sol_data = {}
    for tf in CONFIG["TEST_TIMEFRAMES"]:
        df = fetch_binance_klines(CONFIG["SYMBOL"], tf, CONFIG["BACKTEST_DAYS"])
        if not df.empty:
            sol_data[tf] = df
        time.sleep(1)

    if not sol_data:
        print("❌ No data fetched. Check internet connection.")
        exit()

    # ── Announce on Telegram ──────────────────────────────────────────────────
    send_msg(
        f"🚀 *SOL/USDT RSI×WMA Evolutionary Backtest*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 *Signal:* RSI crosses above WMA(RSI)\n"
        f"🔧 *RSI:* {CONFIG['GEN_START_RSI']}→"
        f"{CONFIG['GEN_START_RSI']+(CONFIG['TOTAL_GENERATIONS']-1)*CONFIG['GEN_STEP_RSI']}"
        f" (+2/gen) | *WMA:* {CONFIG['WMA_FIXED']} fixed\n"
        f"💰 *Capital:* $1000 USDC | *Risk:* 1%/trade\n"
        f"📉 *SL:* 2×ATR | *TGT:* 2.2R | *Trail:* 1×ATR\n"
        f"⏱ *TFs:* 5m, 15m, 1h\n"
        f"🧬 *10 Gens | 1 trade at a time*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📲 Expect 11 reports (10 gen + 1 master)"
    )
    time.sleep(2)

    all_stats = []

    # ── GENERATION LOOP ───────────────────────────────────────────────────────
    for gen_info in GENERATIONS:
        gen_num = gen_info["gen"]
        rsi_p   = gen_info["rsi_p"]
        wma_p   = gen_info["wma_p"]

        print(f"\n{'='*50}")
        print(f"🧬 GENERATION {gen_num} | RSI({rsi_p}) × WMA({wma_p})")
        print(f"{'='*50}")

        gs = make_gen_stats(gen_num, rsi_p, wma_p)

        # ── TIMEFRAME LOOP ────────────────────────────────────────────────────
        for tf, df in sol_data.items():
            print(f"  📊 SOL/USDT @ {tf} | RSI({rsi_p}) × WMA({wma_p})")

            cerebro = bt.Cerebro()
            cerebro.adddata(
                bt.feeds.PandasData(dataname=df.copy()),
                name=f"SOL_{tf}"
            )
            cerebro.addstrategy(
                RSIWMACrossStrategy,
                rsi_p     = rsi_p,
                wma_p     = wma_p,
                atr_p     = CONFIG["ATR_PERIOD"],
                atr_sl    = CONFIG["ATR_SL_MULT"],
                rr        = CONFIG["RR_RATIO"],
                atr_trail = CONFIG["ATR_TRAIL_MULT"],
            )
            cerebro.broker.setcash(CONFIG["INITIAL_USDC"])
            cerebro.broker.setcommission(commission=0.001)  # 0.1% Binance fee
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tr")
            cerebro.addanalyzer(bt.analyzers.DrawDown,       _name="dd")

            try:
                results = cerebro.run()
                strat   = results[0]
                res     = strat.analyzers.tr.get_analysis()
                dd      = strat.analyzers.dd.get_analysis()

                if 'total' in res and res.total.total > 0:
                    pnl = cerebro.broker.get_value() - CONFIG["INITIAL_USDC"]

                    # Global gen stats
                    gs["profit"]      += pnl
                    gs["trades"]      += res.total.total
                    gs["wins"]        += res.won.total           if 'won'  in res else 0
                    gs["gross_p"]     += res.won.pnl.total       if 'won'  in res else 0
                    gs["gross_l"]     += abs(res.lost.pnl.total) if 'lost' in res else 0
                    gs["max_dd"]       = max(gs["max_dd"], dd.max.drawdown)
                    gs["target_wins"] += strat.target_wins
                    gs["total_loss"]  += strat.total_loss

                    # Per-timeframe stats
                    gs["tf_stats"][tf]["profit"] += pnl
                    gs["tf_stats"][tf]["trades"] += res.total.total
                    gs["tf_stats"][tf]["wins"]   += res.won.total if 'won' in res else 0

                    # Best timeframe
                    if pnl > gs["best_tf"]["profit"]:
                        gs["best_tf"] = {"name": f"SOL/USDT ({tf})", "profit": pnl}

                    print(f"    ✅ P/L: ${pnl:.2f} | Trades: {res.total.total}")

            except Exception as e:
                print(f"    ❌ Error @ {tf}: {e}")

            time.sleep(CONFIG["DELAY_SECONDS"])

        # ── Generation done → send report ─────────────────────────────────────
        all_stats.append(gs)
        send_gen_report(gs)
        print(f"\n✅ Generation {gen_num} report sent to Telegram.\n")
        time.sleep(2)

    # ── All 10 generations done → Master Report ───────────────────────────────
    print("\n🏛 Sending Master Report...\n")
    send_master_report(all_stats)
    print("🏁 All done.")
