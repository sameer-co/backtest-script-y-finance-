import backtrader as bt
import yfinance as yf
import pandas as pd
import requests
import time
import collections
import numpy as np

# ── Compatibility Patch ───────────────────────────────────────────────────────
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "TOKEN"            : "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg",
    "CHAT_ID"          : "1950462171",
    "INITIAL_CASH"     : 50000,
    "RISK_PERCENT"     : 0.02,
    "BACKTEST_PERIOD"  : "2y",
    "WATCHLIST_FILE"   : "watchlist.txt",
    "TEST_TIMEFRAMES"  : ["2h", "4h"],
    "DELAY_SECONDS"    : 5,

    # ── MA EVOLUTION SETTINGS ──────────────────────────────────────────────
    # Gen 1 starts at MA period 9, each gen adds +2
    # Gen 1: MA=9 | Gen 2: MA=11 | ... | Gen 10: MA=27
    "GEN_START_MA"     : 9,
    "GEN_STEP"         : 2,
    "TOTAL_GENERATIONS": 10,

    # ── TOGGLE: Which MA types to test ────────────────────────────────────
    # Options: ["EMA"], ["SMA"], ["EMA", "SMA"]
    "MA_TYPES"         : ["EMA", "SMA"],

    # ── ATR SETTINGS ──────────────────────────────────────────────────────
    "ATR_PERIOD"       : 14,
    "ATR_SL_MULTIPLIER": 2.0,   # SL = entry - (2 × ATR)
    "RR_RATIO"         : 2.5,   # Target = entry + (SL_distance × RR)
    "ATR_TRAIL_MULT"   : 2.0,   # Trailing stop multiplier after half-exit
}

# ── Pre-compute generation parameter table ────────────────────────────────────
GENERATIONS = [
    {
        "gen"  : g + 1,
        "ma_p" : CONFIG["GEN_START_MA"] + (g * CONFIG["GEN_STEP"]),
    }
    for g in range(CONFIG["TOTAL_GENERATIONS"])
]


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY  — Close price crosses above EMA/SMA → BUY
#              SL = entry − (ATR_MULT × ATR)
#              Target = entry + (SL_dist × RR)
#              Partial exit at target, trailing stop on remainder
# ══════════════════════════════════════════════════════════════════════════════
class MACrossStrategy(bt.Strategy):
    params = (
        ('ma_p'      , 9),
        ('ma_type'   , 'EMA'),   # 'EMA' or 'SMA'
        ('atr_p'     , 14),
        ('atr_sl'    , 2.0),
        ('rr'        , 2.5),
        ('atr_trail' , 2.0),
    )

    def __init__(self):
        # Select MA indicator based on toggle
        if self.p.ma_type == 'EMA':
            self.ma = bt.indicators.EMA(self.data.close, period=self.p.ma_p)
        else:
            self.ma = bt.indicators.SMA(self.data.close, period=self.p.ma_p)

        self.atr        = bt.indicators.ATR(self.data, period=self.p.atr_p)
        self.stop_loss  = None
        self.target     = None
        self.half_booked = False

        # Crossover detector: close crosses above MA
        self.crossover = bt.indicators.CrossOver(self.data.close, self.ma)

    def next(self):
        if not self.position:
            # Entry: close price crosses above MA (crossover == 1)
            if self.crossover[0] == 1:
                entry       = self.data.close[0]
                sl_distance = self.p.atr_sl * self.atr[0]   # 2 × ATR
                stop        = entry - sl_distance
                target      = entry + (sl_distance * self.p.rr)

                if sl_distance > 0:
                    qty       = int((self.broker.get_value() * CONFIG["RISK_PERCENT"]) / sl_distance)
                    final_qty = min(qty, int(self.broker.get_cash() / entry))
                    if final_qty > 0:
                        self.buy(size=final_qty)
                        self.stop_loss   = stop
                        self.target      = target
                        self.half_booked = False

        elif self.position:
            # Partial profit at target → sell half
            if not self.half_booked and self.data.high[0] >= self.target:
                self.sell(size=int(self.position.size / 2))
                self.half_booked = True
                self.stop_loss   = max(self.stop_loss, self.data.open[0])

            # Trailing stop on remaining position
            if self.half_booked:
                self.stop_loss = max(
                    self.stop_loss,
                    self.data.close[0] - (self.atr[0] * self.p.atr_trail)
                )

            # Hard stop hit → close everything
            if self.data.low[0] <= self.stop_loss:
                self.close()


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


def resample_data(df, timeframe):
    if timeframe == "1h":
        return df
    tf_map = {"1d": "D", "4h": "4h", "2h": "2h"}
    return df.resample(tf_map.get(timeframe, timeframe)).apply(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    ).dropna()


def make_gen_stats(gen_num, ma_p, ma_type):
    return {
        "gen"       : gen_num,
        "ma_p"      : ma_p,
        "ma_type"   : ma_type,
        "profit"    : 0.0,
        "trades"    : 0,
        "wins"      : 0,
        "gross_p"   : 0.0,
        "gross_l"   : 0.0,
        "max_dd"    : 0.0,
        "best_stock": {"name": "", "profit": -999999},
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
        f"🧬 *GENERATION {gs['gen']} REPORT*\n"
        f"🔧 *MA Type:* {gs['ma_type']}({gs['ma_p']})\n"
        f"📌 *Signal:* Close × {gs['ma_type']} | SL: 2×ATR\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *Net P/L:*        ₹{gs['profit']:,.2f}\n"
        f"{wr_emoji} *Win Rate:*      {win_rate:.2f}%\n"
        f"{pf_emoji} *Profit Factor:* {profit_factor:.2f}\n"
        f"📊 *Total Trades:*   {gs['trades']}\n"
        f"✅ *Wins:*           {gs['wins']}\n"
        f"❌ *Losses:*         {losses}\n"
        f"📉 *Max Drawdown:*   {gs['max_dd']:.2f}%\n"
        f"🎯 *Expectancy:*     ₹{expectancy:.2f}/trade\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🌟 *Best Stock:* {gs['best_stock']['name']}\n"
        f"   Profit: ₹{gs['best_stock']['profit']:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━"
    )
    print(report)
    send_msg(report)


# ══════════════════════════════════════════════════════════════════════════════
#  PER MA-TYPE SUMMARY  (after all 10 gens of one MA type finish)
# ══════════════════════════════════════════════════════════════════════════════
def send_matype_summary(all_stats, ma_type):
    best_gen  = max(all_stats, key=lambda x: x["profit"])
    worst_gen = min(all_stats, key=lambda x: x["profit"])

    table = ""
    for gs in all_stats:
        wr = (gs["wins"] / gs["trades"] * 100) if gs["trades"] > 0 else 0
        pf = (gs["gross_p"] / gs["gross_l"])   if gs["gross_l"] > 0 else 0
        arrow = "⭐" if gs["gen"] == best_gen["gen"] else ("🔴" if gs["gen"] == worst_gen["gen"] else "▪️")
        table += (
            f"{arrow} *G{gs['gen']}* {ma_type}{gs['ma_p']} | "
            f"₹{gs['profit']:,.0f} | {wr:.1f}% | PF:{pf:.2f} | T:{gs['trades']}\n"
        )

    total_profit  = sum(g["profit"]  for g in all_stats)
    total_trades  = sum(g["trades"]  for g in all_stats)
    total_wins    = sum(g["wins"]    for g in all_stats)
    total_gross_p = sum(g["gross_p"] for g in all_stats)
    total_gross_l = sum(g["gross_l"] for g in all_stats)
    overall_wr    = (total_wins / total_trades * 100) if total_trades > 0 else 0
    overall_pf    = (total_gross_p / total_gross_l)   if total_gross_l > 0 else 0

    profits = [g["profit"] for g in all_stats]
    trend   = "📈 IMPROVING" if profits[-1] > profits[0] else "📉 DECLINING"
    f5_avg  = np.mean(profits[:5])
    l5_avg  = np.mean(profits[5:])

    summary = (
        f"🏛 *{ma_type} — 10 GENERATION SUMMARY*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 *GENERATION TABLE*\n"
        f"*(P/L | WinRate | PF | Trades)*\n\n"
        f"{table}"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *AGGREGATED STATS*\n"
        f"💰 Total P/L:    ₹{total_profit:,.2f}\n"
        f"✅ Win Rate:      {overall_wr:.2f}%\n"
        f"⚖️ Profit Factor: {overall_pf:.2f}\n"
        f"📊 Total Trades:  {total_trades}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏆 *BEST GEN:* G{best_gen['gen']} → {ma_type}({best_gen['ma_p']}) | ₹{best_gen['profit']:,.2f}\n"
        f"💀 *WORST GEN:* G{worst_gen['gen']} → {ma_type}({worst_gen['ma_p']}) | ₹{worst_gen['profit']:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 *TREND:* {trend}\n"
        f"   First 5 avg: ₹{f5_avg:,.0f} → Last 5 avg: ₹{l5_avg:,.0f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 *BEST PARAM:* {ma_type}({best_gen['ma_p']}) → Gen {best_gen['gen']}\n"
        f"🏁 *{ma_type} Test Complete.*"
    )
    print(summary)
    send_msg(summary)


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER REPORT  — EMA vs SMA head-to-head comparison
# ══════════════════════════════════════════════════════════════════════════════
def send_master_report(all_results):
    """
    all_results = {
        "EMA": [gs_gen1, gs_gen2, ... gs_gen10],
        "SMA": [gs_gen1, gs_gen2, ... gs_gen10],
    }
    """
    msg = (
        f"🏆 *MASTER REPORT — EMA vs SMA*\n"
        f"🧬 *10 Gens | Start MA=9 | Step +2*\n"
        f"📌 *Signal:* Close Crossover | SL: 2×ATR\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    summary_rows = {}
    for ma_type, stats_list in all_results.items():
        best  = max(stats_list, key=lambda x: x["profit"])
        worst = min(stats_list, key=lambda x: x["profit"])
        total_p = sum(g["profit"]  for g in stats_list)
        total_t = sum(g["trades"]  for g in stats_list)
        total_w = sum(g["wins"]    for g in stats_list)
        gp      = sum(g["gross_p"] for g in stats_list)
        gl      = sum(g["gross_l"] for g in stats_list)
        wr      = (total_w / total_t * 100) if total_t > 0 else 0
        pf      = (gp / gl)                 if gl > 0      else 0

        summary_rows[ma_type] = {
            "total_p": total_p, "wr": wr, "pf": pf,
            "total_t": total_t, "best": best, "worst": worst
        }

        msg += (
            f"📌 *{ma_type} OVERVIEW*\n"
            f"   💰 Total P/L:   ₹{total_p:,.2f}\n"
            f"   ✅ Win Rate:    {wr:.2f}%\n"
            f"   ⚖️ Profit Factor:{pf:.2f}\n"
            f"   📊 Trades:      {total_t}\n"
            f"   🏆 Best Gen:    G{best['gen']} {ma_type}({best['ma_p']}) ₹{best['profit']:,.2f}\n"
            f"   💀 Worst Gen:   G{worst['gen']} {ma_type}({worst['ma_p']}) ₹{worst['profit']:,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    # Head-to-head winner
    if len(summary_rows) == 2:
        ema = summary_rows.get("EMA", {})
        sma = summary_rows.get("SMA", {})
        if ema and sma:
            winner    = "EMA" if ema["total_p"] >= sma["total_p"] else "SMA"
            win_diff  = abs(ema["total_p"] - sma["total_p"])
            wr_winner = "EMA" if ema["wr"] >= sma["wr"] else "SMA"
            pf_winner = "EMA" if ema["pf"] >= sma["pf"] else "SMA"

            msg += (
                f"⚔️ *HEAD-TO-HEAD VERDICT*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 Better P/L:       *{winner}* (by ₹{win_diff:,.2f})\n"
                f"✅ Better Win Rate:  *{wr_winner}* "
                f"(EMA:{ema['wr']:.1f}% vs SMA:{sma['wr']:.1f}%)\n"
                f"⚖️ Better PF:        *{pf_winner}* "
                f"(EMA:{ema['pf']:.2f} vs SMA:{sma['pf']:.2f})\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🥇 *OVERALL WINNER: {winner}*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
            )

    msg += (
        f"📅 Period: {CONFIG['BACKTEST_PERIOD']} | "
        f"TFs: {', '.join(CONFIG['TEST_TIMEFRAMES'])}\n"
        f"🏁 *Full Evolutionary Test Complete.*"
    )
    print(msg)
    send_msg(msg)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Load Watchlist ────────────────────────────────────────────────────────
    with open(CONFIG["WATCHLIST_FILE"], "r") as f:
        symbols = [s.strip().upper() for s in f.read().splitlines() if s.strip()]

    # ── Download raw 1h data once for all symbols ─────────────────────────────
    print("📥 Downloading raw data for all symbols...")
    raw_data = {}
    for s in symbols:
        ticker = s if "." in s else f"{s}.NS"
        df = yf.download(ticker, period=CONFIG["BACKTEST_PERIOD"], interval="1h", progress=False)
        if df.empty:
            print(f"  ⚠️  No data for {ticker}, skipping.")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col) for col in df.columns]
        raw_data[ticker] = df
        print(f"  ✅ {ticker}: {len(df)} bars loaded")
        time.sleep(1)

    print(f"\n🚀 Starting Evolutionary Test → MA Types: {CONFIG['MA_TYPES']}\n")

    # Storage: one list of gen stats per MA type
    all_results = {ma_type: [] for ma_type in CONFIG["MA_TYPES"]}

    # ── OUTER LOOP: MA Type (EMA → then SMA) ─────────────────────────────────
    for ma_type in CONFIG["MA_TYPES"]:

        print(f"\n{'#'*55}")
        print(f"  🔁 Starting {ma_type} — 10 Generation Run")
        print(f"{'#'*55}")

        # Announce start of this MA type on Telegram
        send_msg(
            f"🔁 *Starting {ma_type} Evolutionary Test*\n"
            f"🧬 10 Generations | MA start=9 | Step +2\n"
            f"📌 Signal: Close × {ma_type} | SL: 2×ATR"
        )
        time.sleep(2)

        # ── INNER LOOP: Generations 1–10 ─────────────────────────────────────
        for gen_info in GENERATIONS:
            gen_num = gen_info["gen"]
            ma_p    = gen_info["ma_p"]

            print(f"\n{'='*50}")
            print(f"🧬 {ma_type} | GENERATION {gen_num} | MA Period = {ma_p}")
            print(f"{'='*50}")

            gs = make_gen_stats(gen_num, ma_p, ma_type)

            # ── Loop: Symbols × Timeframes ────────────────────────────────
            for ticker, raw_df in raw_data.items():
                for tf in CONFIG["TEST_TIMEFRAMES"]:
                    print(f"  📊 {ticker} @ {tf} | {ma_type}({ma_p}) | SL=2×ATR")

                    df = resample_data(raw_df.copy(), tf)

                    cerebro = bt.Cerebro()
                    cerebro.adddata(
                        bt.feeds.PandasData(dataname=df),
                        name=f"{ticker}_{tf}"
                    )
                    cerebro.addstrategy(
                        MACrossStrategy,
                        ma_p      = ma_p,
                        ma_type   = ma_type,
                        atr_p     = CONFIG["ATR_PERIOD"],
                        atr_sl    = CONFIG["ATR_SL_MULTIPLIER"],
                        rr        = CONFIG["RR_RATIO"],
                        atr_trail = CONFIG["ATR_TRAIL_MULT"],
                    )
                    cerebro.broker.setcash(CONFIG["INITIAL_CASH"])
                    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tr")
                    cerebro.addanalyzer(bt.analyzers.DrawDown,       _name="dd")

                    try:
                        results = cerebro.run()
                        res = results[0].analyzers.tr.get_analysis()
                        dd  = results[0].analyzers.dd.get_analysis()

                        if 'total' in res and res.total.total > 0:
                            pnl = cerebro.broker.get_value() - CONFIG["INITIAL_CASH"]

                            gs["profit"]  += pnl
                            gs["trades"]  += res.total.total
                            gs["wins"]    += res.won.total        if 'won'  in res else 0
                            gs["gross_p"] += res.won.pnl.total    if 'won'  in res else 0
                            gs["gross_l"] += abs(res.lost.pnl.total) if 'lost' in res else 0
                            gs["max_dd"]   = max(gs["max_dd"], dd.max.drawdown)

                            if pnl > gs["best_stock"]["profit"]:
                                gs["best_stock"] = {"name": f"{ticker} ({tf})", "profit": pnl}

                    except Exception as e:
                        print(f"    ❌ Error on {ticker} @ {tf}: {e}")

                time.sleep(CONFIG["DELAY_SECONDS"])

            # ── Generation complete → send per-gen report ─────────────────
            all_results[ma_type].append(gs)
            send_gen_report(gs)
            print(f"\n✅ {ma_type} Generation {gen_num} report sent.\n")
            time.sleep(2)

        # ── All 10 gens of this MA type done → send MA-type summary ──────────
        send_matype_summary(all_results[ma_type], ma_type)
        print(f"\n✅ {ma_type} 10-Generation Summary sent.\n")
        time.sleep(3)

    # ── ALL MA TYPES DONE → Send Master EMA vs SMA Report ────────────────────
    print("\n🏆 Sending Master EMA vs SMA Report...\n")
    send_master_report(all_results)
    print("🏁 All done.")
