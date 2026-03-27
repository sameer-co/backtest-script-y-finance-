import backtrader as bt
import yfinance as yf
import pandas as pd
import os
import requests
import time
import collections

# Compatibility Patch for Python 3.10+
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable

# --- CUSTOMIZABLE PARAMETERS ---
CONFIG = {
    "TOKEN": "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg",
    "CHAT_ID": "1950462171",
    "INITIAL_CASH": 50000,
    "RISK_PERCENT": 0.02,
    "RSI_PERIOD": 40,
    "WMA_PERIOD": 15,
    "ATR_PERIOD": 14,
    "TARGET_RR": 2.5,
    "ATR_TRAIL_MULT": 2.0,
    "BACKTEST_PERIOD": "2y",
    "WATCHLIST_FILE": "backtest.txt",
    "TEST_TIMEFRAMES": ["1h", "4h", "1d"], # ADD/REMOVE TIMEFRAMES HERE
    "DELAY_SECONDS": 5 
}

class AdvancedRSIStrategy(bt.Strategy):
    params = (
        ('rsi_p', CONFIG["RSI_PERIOD"]),
        ('wma_p', CONFIG["WMA_PERIOD"]),
        ('atr_p', CONFIG["ATR_PERIOD"]),
        ('rr', CONFIG["TARGET_RR"]),
        ('atr_m', CONFIG["ATR_TRAIL_MULT"]),
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
            if self.rsi[0] > self.wma_rsi[0] and self.rsi[-1] <= self.wma_rsi[-1]:
                entry = self.data.close[0]
                yest_low = self.data.low[-1]
                risk_per_share = entry - yest_low

                if risk_per_share > 0:
                    risk_amount = self.broker.get_value() * CONFIG["RISK_PERCENT"]
                    qty = int(risk_amount / risk_per_share)
                    max_qty = int(self.broker.get_cash() / entry)
                    final_qty = min(qty, max_qty)

                    if final_qty > 0:
                        self.buy(size=final_qty)
                        self.stop_loss = yest_low
                        self.target = entry + (risk_per_share * self.p.rr)
                        self.half_booked = False

        elif self.position:
            if not self.half_booked and self.data.high[0] >= self.target:
                self.sell(size=int(self.position.size / 2))
                self.half_booked = True
                self.stop_loss = max(self.stop_loss, self.data.open[0])

            if self.half_booked:
                current_atr_trail = self.data.close[0] - (self.atr[0] * self.p.atr_m)
                self.stop_loss = max(self.stop_loss, current_atr_trail)

            if self.data.low[0] <= self.stop_loss:
                self.close()

def send_msg(text):
    url = f"https://api.telegram.org/bot{CONFIG['TOKEN']}/sendMessage"
    payload = {"chat_id": CONFIG["CHAT_ID"], "text": text, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=10)
    except: pass

def resample_data(df, timeframe):
    if timeframe == "1h": return df
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    tf_map = {"1d": "D", "4h": "4h", "2h": "2h"}
    return df.resample(tf_map.get(timeframe, timeframe)).apply(logic).dropna()

if __name__ == "__main__":
    if not os.path.exists(CONFIG["WATCHLIST_FILE"]):
        print(f"Error: {CONFIG['WATCHLIST_FILE']} not found!")
        exit()

    with open(CONFIG["WATCHLIST_FILE"], "r") as f:
        symbols = [s.strip().upper() for s in f.read().splitlines() if s.strip()]

    total_portfolio_pl = 0

    for s in symbols:
        ticker = s if "." in s else f"{s}.NS"
        print(f"📥 Downloading 2y hourly data for {ticker}...")
        
        # Download 1h as base (highest resolution for 2y)
        raw_df = yf.download(ticker, period="2y", interval="1h", progress=False)
        if raw_df.empty: continue

        # Fix Multi-index Columns
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        raw_df.columns = [str(col) for col in raw_df.columns]

        for tf in CONFIG["TEST_TIMEFRAMES"]:
            print(f"⌛ Testing {ticker} on {tf}...")
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
                    net_profit = cerebro.broker.get_value() - CONFIG["INITIAL_CASH"]
                    total_portfolio_pl += net_profit
                    
                    summary = (
                        f"📊 *Backtest: {ticker}*\n"
                        f"🕒 *Timeframe:* `{tf}`\n"
                        f"Trades: {res.total.total} | Wins: {res.won.total if 'won' in res else 0}\n"
                        f"🔥 Max DD: {dd.max.drawdown:.2f}%\n"
                        f"💰 Net P/L: ₹{net_profit:.2f}"
                    )
                    send_msg(summary)
                    print(f"✅ Sent {tf} result. Sleeping {CONFIG['DELAY_SECONDS']}s...")
                    time.sleep(CONFIG["DELAY_SECONDS"])
            except Exception as e:
                print(f"Error on {ticker} {tf}: {e}")

    # FINAL TOTAL
    send_msg(f"🏁 *BACKTEST COMPLETE*\n\n💼 *Total Portfolio P/L:* ₹{total_portfolio_pl:.2f}")
    print("All tests finished.")
