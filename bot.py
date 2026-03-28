import backtrader as bt
import yfinance as yf
import pandas as pd
import os
import requests
import time
import collections
import numpy as np

# --- COMPATIBILITY & CONFIG ---
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable

CONFIG = {
    "TOKEN": "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg",
    "CHAT_ID": "1950462171",
    "INITIAL_CASH": 50000,
    "RISK_PERCENT": 0.02,
    "EMA_P": 13,                # Fast Signal
    "SMA_P": 18,                # Slow Signal
    "ATR_PERIOD": 14,
    "TARGET_RR": 2.5,
    "ATR_TRAIL_MULT": 2.0,
    "BACKTEST_PERIOD": "2y",
    "WATCHLIST_FILE": "backtest_list.txt",
    "TEST_TIMEFRAMES": ["1h", "4h", "1d"],
    "DELAY_SECONDS": 5 
}

# --- GLOBAL TRACKER FOR DEEP REPORT ---
STATS = {
    "global": {"profit": 0, "trades": 0, "wins": 0, "gross_p": 0, "gross_l": 0, "max_dd": 0},
    "timeframes": {tf: {"profit": 0, "trades": 0, "wins": 0} for tf in CONFIG["TEST_TIMEFRAMES"]},
    "best": {"name": "", "profit": -999999}
}

# --- CUSTOM INDICATOR: POSITIVE VOLUME INDEX ---
class PositiveVolumeIndex(bt.Indicator):
    lines = ('pvi',)
    def __init__(self):
        self.addminperiod(2)

    def next(self):
        if len(self) == 1:
            self.lines.pvi[0] = 1000.0
            return

        prev_pvi = self.lines.pvi[-1]
        # PVI updates only if today's volume > yesterday's volume
        if self.data.volume[0] > self.data.volume[-1]:
            change = (self.data.close[0] - self.data.close[-1]) / self.data.close[-1]
            self.lines.pvi[0] = prev_pvi + (change * prev_pvi)
        else:
            self.lines.pvi[0] = prev_pvi

# --- STRATEGY: PVI CROSSOVER MODEL ---
class PVI_Strategy(bt.Strategy):
    params = (
        ('ema_p', CONFIG["EMA_P"]),
        ('sma_p', CONFIG["SMA_P"]),
        ('atr_p', CONFIG["ATR_PERIOD"]),
        ('rr', CONFIG["TARGET_RR"]),
        ('atr_m', CONFIG["ATR_TRAIL_MULT"]),
    )

    def __init__(self):
        self.pvi = PositiveVolumeIndex(self.data)
        self.fast_ema = bt.indicators.EMA(self.pvi, period=self.p.ema_p)
        self.slow_sma = bt.indicators.SMA(self.pvi, period=self.p.sma_p)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_p)
        
        self.stop_loss = None
        self.target = None
        self.half_booked = False

    def next(self):
        # 1. ENTRY MODEL (13 EMA > 18 SMA of PVI)
        if not self.position:
            if self.fast_ema[0] > self.slow_sma[0] and self.fast_ema[-1] <= self.slow_sma[-1]:
                entry = self.data.close[0]
                risk_per_share = entry - self.data.low[-1]

                if risk_per_share > 0:
                    risk_amount = self.broker.get_value() * CONFIG["RISK_PERCENT"]
                    qty = int(risk_amount / risk_per_share)
                    max_qty = int(self.broker.get_cash() / entry)
                    final_qty = min(qty, max_qty)

                    if final_qty > 0:
                        self.buy(size=final_qty)
                        self.stop_loss = self.data.low[-1]
                        self.target = entry + (risk_per_share * self.p.rr)
                        self.half_booked = False

        # 2. EXIT MODEL (Same Professional Risk Mgmt)
        elif self.position:
            # Target Hit (Book 50%)
            if not self.half_booked and self.data.high[0] >= self.target:
                self.sell(size=int(self.position.size / 2))
                self.half_booked = True
                self.stop_loss = max(self.stop_loss, self.data.open[0])

            # ATR Trailing Stop (Active after half booked)
            if self.half_booked:
                current_trail = self.data.close[0] - (self.atr[0] * self.p.atr_m)
                self.stop_loss = max(self.stop_loss, current_trail)

            # Hard Stop Loss
            if self.data.low[0] <= self.stop_loss:
                self.close()

# --- UTILS ---
def send_msg(text):
    url = f"https://api.telegram.org/bot{CONFIG['TOKEN']}/sendMessage"
    payload = {"chat_id": CONFIG["CHAT_ID"], "text": text, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=10)
    except: pass

def resample_data(df, timeframe):
    if timeframe == "1h": return df
    tf_map = {"1d": "D", "4h": "4h", "2h": "2h"}
    logic = {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
    return df.resample(tf_map.get(timeframe, timeframe)).apply(logic).dropna()

# --- EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(CONFIG["WATCHLIST_FILE"]):
        print("Watchlist file not found!")
        exit()

    with open(CONFIG["WATCHLIST_FILE"], "r") as f:
        symbols = [s.strip().upper() for s in f.read().splitlines() if s.strip()]

    for s in symbols:
        ticker = s if "." in s else f"{s}.NS"
        print(f"📥 Fetching data for {ticker}...")
        
        # Pull 2 years of 1h data as the foundation
        raw_df = yf.download(ticker, period="2y", interval="1h", progress=False)
        if raw_df.empty: continue
        
        # Data Cleaning for Backtrader
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        raw_df.columns = [str(col) for col in raw_df.columns]

        for tf in CONFIG["TEST_TIMEFRAMES"]:
            print(f"⌛ Auditing {ticker} on {tf}...")
            df = resample_data(raw_df.copy(), tf)
            
            cerebro = bt.Cerebro()
            cerebro.adddata(bt.feeds.PandasData(dataname=df), name=f"{ticker}_{tf}")
            cerebro.addstrategy(PVI_Strategy)
            cerebro.broker.setcash(CONFIG["INITIAL_CASH"])
            
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tr")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

            try:
                results = cerebro.run()
                res = results[0].analyzers.tr.get_analysis()
                dd = results[0].analyzers.dd.get_analysis()

                if 'total' in res and res.total.total > 0:
                    pnl = cerebro.broker.get_value() - CONFIG["INITIAL_CASH"]
                    
                    # Update Deep Analytics
                    STATS["global"]["profit"] += pnl
                    STATS["global"]["trades"] += res.total.total
                    STATS["global"]["wins"] += res.won.total if 'won' in res else 0
                    STATS["global"]["gross_p"] += res.won.pnl.total if 'won' in res else 0
                    STATS["global"]["gross_l"] += abs(res.lost.pnl.total) if 'lost' in res else 0
                    STATS["global"]["max_dd"] = max(STATS["global"]["max_dd"], dd.max.drawdown)
                    
                    STATS["timeframes"][tf]["profit"] += pnl
                    STATS["timeframes"][tf]["trades"] += res.total.total
                    STATS["timeframes"][tf]["wins"] += res.won.total if 'won' in res else 0

                    if pnl > STATS["best"]["profit"]:
                        STATS["best"] = {"name": f"{ticker} ({tf})", "profit": pnl}

                    # Instant Update
                    send_msg(f"✅ {ticker} ({tf}): ₹{pnl:.2f}")
                    time.sleep(CONFIG["DELAY_SECONDS"])

            except Exception as e:
                print(f"Error skipping {ticker} {tf}: {e}")

    # --- THE FINAL PROFESSIONAL AUDIT ---
    g = STATS["global"]
    win_rate = (g["wins"] / g["trades"] * 100) if g["trades"] > 0 else 0
    pf = (g["gross_p"] / g["gross_l"]) if g["gross_l"] > 0 else 0
    expectancy = (g["profit"] / g["trades"]) if g["trades"] > 0 else 0

    tf_results = ""
    for tf, data in STATS["timeframes"].items():
        acc = (data["wins"]/data["trades"]*100) if data["trades"] > 0 else 0
        tf_results += f"• *{tf}:* ₹{data['profit']:.0f} (Acc: {acc:.1f}%)\n"

    final_report = (
        f"🏛 *PVI STRATEGY AUDIT REPORT*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"💰 *NET PORTFOLIO P/L:* ₹{g['profit']:.2f}\n"
        f"📊 *OVERALL ACCURACY:* {win_rate:.2f}%\n"
        f"⚖️ *PROFIT FACTOR:* {pf:.2f}\n"
        f"📉 *MAX SYSTEM DRAWDOWN:* {g['max_dd']:.2f}%\n"
        f"🎯 *EXPECTANCY:* ₹{expectancy:.2f} / trade\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"⏳ *TIMEFRAME PERFORMANCE*\n"
        f"{tf_results}"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🌟 *STAR PERFORMER:* {STATS['best']['name']}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📅 *PERIOD:* 2 Years | *TOTAL TRADES:* {g['trades']}\n"
        f"🏁 *Deep Audit Complete.*"
    )
    send_msg(final_report)
