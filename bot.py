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
    "DELAY_SECONDS": 5 
}

# --- ADVANCED STATS TRACKER ---
STATS = {
    "global": {"profit": 0, "trades": 0, "wins": 0, "gross_p": 0, "gross_l": 0, "max_dd": 0},
    "timeframes": {tf: {"profit": 0, "trades": 0, "wins": 0} for tf in CONFIG["TEST_TIMEFRAMES"]},
    "best": {"name": "", "profit": -999999}
}

class AdvancedRSIStrategy(bt.Strategy):
    params = (
        ('rsi_p', 40), 
        ('wma_p', 15), 
        ('atr_p', 14), 
        ('rr', 2.5), 
        ('atr_m', 2.0),
        ('adx_p', 14),     # Added ADX Period
        ('adx_limit', 25)  # Added ADX Threshold
    )
    
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_p)
        self.wma_rsi = bt.indicators.WMA(self.rsi, period=self.p.wma_p)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_p)
        self.adx = bt.indicators.ADX(self.data, period=self.p.adx_p) # Added ADX
        self.stop_loss = None
        self.target = None
        self.half_booked = False

    def next(self):
        if not self.position:
            # Updated Logic: RSI crossover AND ADX >= 25
            rsi_cross = self.rsi[0] > self.wma_rsi[0] and self.rsi[-1] <= self.wma_rsi[-1]
            trend_strong = self.adx[0] >= self.p.adx_limit
            
            if rsi_cross and trend_strong:
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
            if not self.half_booked and self.data.high[0] >= self.target:
                self.sell(size=int(self.position.size / 2))
                self.half_booked = True
                self.stop_loss = max(self.stop_loss, self.data.open[0])
            if self.half_booked:
                self.stop_loss = max(self.stop_loss, self.data.close[0] - (self.atr[0] * self.p.atr_m))
            if self.data.low[0] <= self.stop_loss:
                self.close()

def send_msg(text):
    url = f"https://api.telegram.org/bot{CONFIG['TOKEN']}/sendMessage"
    try: requests.post(url, json={"chat_id": CONFIG["CHAT_ID"], "text": text, "parse_mode": "Markdown"}, timeout=10)
    except: pass

def resample_data(df, timeframe):
    if timeframe == "1h": return df
    tf_map = {"1d": "D", "4h": "4h", "2h": "2h"}
    return df.resample(tf_map.get(timeframe, timeframe)).apply({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

if __name__ == "__main__":
    if not os.path.exists(CONFIG["WATCHLIST_FILE"]):
        print(f"Error: {CONFIG['WATCHLIST_FILE']} not found.")
    else:
        with open(CONFIG["WATCHLIST_FILE"], "r") as f:
            symbols = [s.strip().upper() for s in f.read().splitlines() if s.strip()]

        for s in symbols:
            ticker = s if "." in s else f"{s}.NS"
            raw_df = yf.download(ticker, period=CONFIG["BACKTEST_PERIOD"], interval="1h", progress=False)
            if raw_df.empty: continue
            if isinstance(raw_df.columns, pd.MultiIndex): raw_df.columns = raw_df.columns.get_level_values(0)
            raw_df.columns = [str(col) for col in raw_df.columns]

            for tf in CONFIG["TEST_TIMEFRAMES"]:
                print(f"Analyzing {ticker} @ {tf}...")
                df = resample_data(raw_df.copy(), tf)
                cerebro = bt.Cerebro()
                cerebro.adddata(bt.feeds.PandasData(dataname=df), name=f"{ticker}_{tf}")
                cerebro.addstrategy(AdvancedRSIStrategy)
                cerebro.broker.setcash(CONFIG["INITIAL_CASH"])
                cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tr")
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
                
                try:
                    results = cerebro.run()
                    res = results[0].analyzers.tr.get_analysis()
                    dd = results[0].analyzers.dd.get_analysis()

                    if 'total' in res and res.total.total > 0:
                        pnl = cerebro.broker.get_value() - CONFIG["INITIAL_CASH"]
                        
                        # Log Global Stats
                        STATS["global"]["profit"] += pnl
                        STATS["global"]["trades"] += res.total.total
                        STATS["global"]["wins"] += res.won.total if 'won' in res else 0
                        STATS["global"]["gross_p"] += res.won.pnl.total if 'won' in res else 0
                        STATS["global"]["gross_l"] += abs(res.lost.pnl.total) if 'lost' in res else 0
                        STATS["global"]["max_dd"] = max(STATS["global"]["max_dd"], dd.max.drawdown)
                        
                        # Log Timeframe Specifics
                        STATS["timeframes"][tf]["profit"] += pnl
                        STATS["timeframes"][tf]["trades"] += res.total.total
                        STATS["timeframes"][tf]["wins"] += res.won.total if 'won' in res else 0

                        if pnl > STATS["best"]["profit"]:
                            STATS["best"] = {"name": f"{ticker} ({tf})", "profit": pnl}

                except Exception as e: print(f"Error analyzing {ticker}: {e}")
            time.sleep(1) # Interval delay

        # --- FINANCIAL ANALYSIS CALCULATION ---
        g = STATS["global"]
        win_rate = (g["wins"] / g["trades"] * 100) if g["trades"] > 0 else 0
        profit_factor = (g["gross_p"] / g["gross_l"]) if g["gross_l"] > 0 else 0
        expectancy = (g["profit"] / g["trades"]) if g["trades"] > 0 else 0

        # Build Timeframe Breakdown String
        tf_report = ""
        for tf, data in STATS["timeframes"].items():
            tf_acc = (data["wins"]/data["trades"]*100) if data["trades"] > 0 else 0
            tf_report += f"• *{tf}:* ₹{data['profit']:.0f} (Acc: {tf_acc:.1f}%)\n"

        # --- THE FINAL MASTER REPORT ---
        master_report = (
            f"🏛 *FINANCIAL PERFORMANCE AUDIT*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 *NET PORTFOLIO P/L:* ₹{g['profit']:.2f}\n"
            f"📊 *OVERALL ACCURACY:* {win_rate:.2f}%\n"
            f"⚖️ *PROFIT FACTOR:* {profit_factor:.2f}\n"
            f"📉 *MAX SYSTEM DRAWDOWN:* {g['max_dd']:.2f}%\n"
            f"🎯 *EXPECTANCY:* ₹{expectancy:.2f} / trade\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⏳ *TIMEFRAME BREAKDOWN*\n"
            f"{tf_report}"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🌟 *STAR PERFORMER:* {STATS['best']['name']}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📅 *PERIOD:* {CONFIG['BACKTEST_PERIOD']} | *TRADES:* {g['trades']}\n"
            f"🏁 *Report Finalized.*"
        )
        send_msg(master_report)
