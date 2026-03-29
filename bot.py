import backtrader as bt
import yfinance as yf
import pandas as pd
import os
import requests
import time
import collections
import numpy as np

# Compatibility Patch
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable

CONFIG = {
    "TOKEN": "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg",
    "CHAT_ID": "1950462171",
    "INITIAL_CASH": 50000,
    "RISK_PERCENT": 0.02,
    "BACKTEST_PERIOD": "2y",
    "WATCHLIST_FILE": "watchlist.txt",
    "TEST_TIMEFRAMES": ["2h", "4h"],
    "DELAY_SECONDS": 5,

    # --- GENERATION CONFIG ---
    # Gen 1 starts at RSI=14, WMA=9
    # Each generation adds +2 to both
    # Gen 1: RSI=14, WMA=9
    # Gen 2: RSI=16, WMA=11
    # ...
    # Gen 10: RSI=32, WMA=27
    "GEN_START_RSI": 14,
    "GEN_START_WMA": 9,
    "GEN_STEP": 2,
    "TOTAL_GENERATIONS": 10,
}

# ── GENERATION PARAMETER TABLE ──────────────────────────────────────────────
# Pre-compute all 10 generation parameters upfront for clarity
GENERATIONS = []
for g in range(CONFIG["TOTAL_GENERATIONS"]):
    GENERATIONS.append({
        "gen": g + 1,
        "rsi_p": CONFIG["GEN_START_RSI"] + (g * CONFIG["GEN_STEP"]),
        "wma_p": CONFIG["GEN_START_WMA"] + (g * CONFIG["GEN_STEP"]),
    })

# ── STATS STRUCTURE ─────────────────────────────────────────────────────────
# One stats block per generation
ALL_GEN_STATS = []


def make_gen_stats(gen_num, rsi_p, wma_p):
    return {
        "gen": gen_num,
        "rsi_p": rsi_p,
        "wma_p": wma_p,
        "profit": 0.0,
        "trades": 0,
        "wins": 0,
        "gross_p": 0.0,
        "gross_l": 0.0,
        "max_dd": 0.0,
        "best_stock": {"name": "", "profit": -999999},
    }


# ── STRATEGY ─────────────────────────────────────────────────────────────────
class AdvancedRSIStrategy(bt.Strategy):
    params = (
        ('rsi_p', 14),
        ('wma_p', 9),
        ('atr_p', 14),
        ('rr', 2.5),
        ('atr_m', 2.0),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_p)
        self.wma_rsi = bt.indicators.WMA(self.rsi, period=self.p.wma_p)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_p)
        self.stop_loss = None
        self.target = None
        self.half_booked = False

    def next(self):
        if not self.position:
            # Entry: RSI crosses above its own WMA
            if self.rsi[0] > self.wma_rsi[0] and self.rsi[-1] <= self.wma_rsi[-1]:
                entry = self.data.close[0]
                risk = entry - self.data.low[-1]
                if risk > 0:
                    qty = int((self.broker.get_value() * CONFIG["RISK_PERCENT"]) / risk)
                    final_qty = min(qty, int(self.broker.get_cash() / entry))
                    if final_qty > 0:
                        self.buy(size=final_qty)
                        self.stop_loss = self.data.low[-1]
                        self.target = entry + (risk * self.p.rr)
                        self.half_booked = False

        elif self.position:
            # Partial profit at target
            if not self.half_booked and self.data.high[0] >= self.target:
                self.sell(size=int(self.position.size / 2))
                self.half_booked = True
                self.stop_loss = max(self.stop_loss, self.data.open[0])

            # Trailing stop after partial exit
            if self.half_booked:
                self.stop_loss = max(
                    self.stop_loss,
                    self.data.close[0] - (self.atr[0] * self.p.atr_m)
                )

            # Hard stop
            if self.data.low[0] <= self.stop_loss:
                self.close()


# ── TELEGRAM ─────────────────────────────────────────────────────────────────
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


# ── DATA HELPERS ──────────────────────────────────────────────────────────────
def resample_data(df, timeframe):
    if timeframe == "1h":
        return df
    tf_map = {"1d": "D", "4h": "4h", "2h": "2h"}
    return df.resample(tf_map.get(timeframe, timeframe)).apply(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    ).dropna()


# ── PER-GENERATION TELEGRAM REPORT ───────────────────────────────────────────
def send_gen_report(gs):
    """Send a detailed report for a single completed generation."""
    win_rate    = (gs["wins"] / gs["trades"] * 100) if gs["trades"] > 0 else 0
    profit_factor = (gs["gross_p"] / gs["gross_l"]) if gs["gross_l"] > 0 else 0
    expectancy  = (gs["profit"] / gs["trades"]) if gs["trades"] > 0 else 0

    # Profit factor emoji
    pf_emoji = "🟢" if profit_factor >= 1.5 else ("🟡" if profit_factor >= 1.0 else "🔴")
    wr_emoji = "🟢" if win_rate >= 50 else "🔴"

    report = (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧬 *GENERATION {gs['gen']} REPORT*\n"
        f"🔧 *Params:* RSI({gs['rsi_p']}) | WMA({gs['wma_p']})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *Net P/L:*        ₹{gs['profit']:,.2f}\n"
        f"{wr_emoji} *Win Rate:*      {win_rate:.2f}%\n"
        f"{pf_emoji} *Profit Factor:* {profit_factor:.2f}\n"
        f"📊 *Total Trades:*   {gs['trades']}\n"
        f"✅ *Wins:*           {gs['wins']}\n"
        f"❌ *Losses:*         {gs['trades'] - gs['wins']}\n"
        f"📉 *Max Drawdown:*   {gs['max_dd']:.2f}%\n"
        f"🎯 *Expectancy:*     ₹{expectancy:.2f}/trade\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🌟 *Best Stock:* {gs['best_stock']['name']}\n"
        f"   Profit: ₹{gs['best_stock']['profit']:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━"
    )
    print(report)
    send_msg(report)


# ── MASTER SUMMARY REPORT ────────────────────────────────────────────────────
def send_master_report(all_stats):
    """Compare all 10 generations and send a master Telegram report."""

    # Find best and worst generations by net profit
    best_gen  = max(all_stats, key=lambda x: x["profit"])
    worst_gen = min(all_stats, key=lambda x: x["profit"])

    # Build generation comparison table
    table = ""
    for gs in all_stats:
        wr = (gs["wins"] / gs["trades"] * 100) if gs["trades"] > 0 else 0
        pf = (gs["gross_p"] / gs["gross_l"]) if gs["gross_l"] > 0 else 0
        arrow = "⭐" if gs["gen"] == best_gen["gen"] else ("🔴" if gs["gen"] == worst_gen["gen"] else "▪️")
        table += (
            f"{arrow} *G{gs['gen']}* RSI{gs['rsi_p']}/WMA{gs['wma_p']} | "
            f"₹{gs['profit']:,.0f} | {wr:.1f}% | PF:{pf:.2f} | T:{gs['trades']}\n"
        )

    # Overall aggregated metrics
    total_profit = sum(g["profit"] for g in all_stats)
    total_trades = sum(g["trades"] for g in all_stats)
    total_wins   = sum(g["wins"] for g in all_stats)
    total_gross_p = sum(g["gross_p"] for g in all_stats)
    total_gross_l = sum(g["gross_l"] for g in all_stats)
    overall_wr   = (total_wins / total_trades * 100) if total_trades > 0 else 0
    overall_pf   = (total_gross_p / total_gross_l) if total_gross_l > 0 else 0

    # Trend: is profit growing or shrinking across generations?
    profits = [g["profit"] for g in all_stats]
    trend_direction = "📈 IMPROVING" if profits[-1] > profits[0] else "📉 DECLINING"
    # Simple slope check across first and last half
    first_half_avg = np.mean(profits[:5])
    second_half_avg = np.mean(profits[5:])
    trend_detail = f"First 5 avg: ₹{first_half_avg:,.0f} → Last 5 avg: ₹{second_half_avg:,.0f}"

    master = (
        f"🏛 *MASTER EVOLUTIONARY REPORT*\n"
        f"🧬 *10 Generations | RSI+WMA Step +2 Each*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 *GENERATION COMPARISON*\n"
        f"*(P/L | WinRate | ProfitFactor | Trades)*\n\n"
        f"{table}"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *OVERALL AGGREGATED STATS*\n"
        f"💰 Total P/L:     ₹{total_profit:,.2f}\n"
        f"✅ Win Rate:       {overall_wr:.2f}%\n"
        f"⚖️ Profit Factor:  {overall_pf:.2f}\n"
        f"📊 Total Trades:   {total_trades}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏆 *BEST GENERATION:*\n"
        f"   Gen {best_gen['gen']} → RSI({best_gen['rsi_p']}) WMA({best_gen['wma_p']})\n"
        f"   P/L: ₹{best_gen['profit']:,.2f}\n\n"
        f"💀 *WORST GENERATION:*\n"
        f"   Gen {worst_gen['gen']} → RSI({worst_gen['rsi_p']}) WMA({worst_gen['wma_p']})\n"
        f"   P/L: ₹{worst_gen['profit']:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 *PARAMETER TREND:* {trend_direction}\n"
        f"   {trend_detail}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 *INSIGHT:* Optimal params appear to be\n"
        f"   RSI({best_gen['rsi_p']}) + WMA({best_gen['wma_p']}) → Gen {best_gen['gen']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 Period: {CONFIG['BACKTEST_PERIOD']} | TFs: {', '.join(CONFIG['TEST_TIMEFRAMES'])}\n"
        f"🏁 *Evolutionary Test Complete.*"
    )
    print(master)
    send_msg(master)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load watchlist
    with open(CONFIG["WATCHLIST_FILE"], "r") as f:
        symbols = [s.strip().upper() for s in f.read().splitlines() if s.strip()]

    # Pre-download all raw data (1h) once to avoid repeated API calls
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

    print(f"\n🚀 Starting {CONFIG['TOTAL_GENERATIONS']} Generation Evolutionary Test...\n")

    # ── LOOP OVER GENERATIONS ──────────────────────────────────────────────
    for gen_info in GENERATIONS:
        gen_num = gen_info["gen"]
        rsi_p   = gen_info["rsi_p"]
        wma_p   = gen_info["wma_p"]

        print(f"\n{'='*50}")
        print(f"🧬 GENERATION {gen_num} | RSI={rsi_p} | WMA={wma_p}")
        print(f"{'='*50}")

        gs = make_gen_stats(gen_num, rsi_p, wma_p)

        # ── LOOP OVER SYMBOLS & TIMEFRAMES ────────────────────────────────
        for ticker, raw_df in raw_data.items():
            for tf in CONFIG["TEST_TIMEFRAMES"]:
                print(f"  📊 {ticker} @ {tf} | RSI={rsi_p} WMA={wma_p}")

                df = resample_data(raw_df.copy(), tf)

                cerebro = bt.Cerebro()
                cerebro.adddata(bt.feeds.PandasData(dataname=df), name=f"{ticker}_{tf}")
                cerebro.addstrategy(
                    AdvancedRSIStrategy,
                    rsi_p=rsi_p,
                    wma_p=wma_p
                )
                cerebro.broker.setcash(CONFIG["INITIAL_CASH"])
                cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tr")
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

                try:
                    results = cerebro.run()
                    res = results[0].analyzers.tr.get_analysis()
                    dd  = results[0].analyzers.dd.get_analysis()

                    if 'total' in res and res.total.total > 0:
                        pnl = cerebro.broker.get_value() - CONFIG["INITIAL_CASH"]

                        # Accumulate into generation stats
                        gs["profit"]  += pnl
                        gs["trades"]  += res.total.total
                        gs["wins"]    += res.won.total if 'won' in res else 0
                        gs["gross_p"] += res.won.pnl.total if 'won' in res else 0
                        gs["gross_l"] += abs(res.lost.pnl.total) if 'lost' in res else 0
                        gs["max_dd"]   = max(gs["max_dd"], dd.max.drawdown)

                        # Track best stock in this generation
                        if pnl > gs["best_stock"]["profit"]:
                            gs["best_stock"] = {"name": f"{ticker} ({tf})", "profit": pnl}

                except Exception as e:
                    print(f"    ❌ Error on {ticker} @ {tf}: {e}")

            time.sleep(CONFIG["DELAY_SECONDS"])

        # ── GENERATION COMPLETE → SEND REPORT ─────────────────────────────
        ALL_GEN_STATS.append(gs)
        send_gen_report(gs)
        print(f"\n✅ Generation {gen_num} report sent to Telegram.\n")
        time.sleep(2)  # Brief pause between generations

    # ── ALL GENERATIONS DONE → SEND MASTER REPORT ─────────────────────────
    print("\n🏛 Sending Master Evolutionary Summary...\n")
    send_master_report(ALL_GEN_STATS)
    print("🏁 All done.")
