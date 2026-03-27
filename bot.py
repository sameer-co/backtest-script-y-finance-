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
    "WATCHLIST_FILE": "backtest_list.txt",
    "DELAY_SECONDS": 5  # Added for rate limiting
}

# --- STRATEGY ENGINE ---
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

# --- UTILS ---
def send_msg(text):
    url = f"https://api.telegram.org/bot{CONFIG['TOKEN']}/sendMessage"
    payload = {"chat_id": CONFIG["CHAT_ID"], "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram Error: {e}")

# --- MAIN RUNNER ---
if __name__ == "__main__":
    if not os.path.exists(CONFIG["WATCHLIST_FILE"]):
        print("Watchlist file missing.")
        exit()

    with open(CONFIG["WATCHLIST_FILE"], "r") as f:
        symbols = [s.strip().upper() for s in f.read().splitlines() if s.strip()]

    total_portfolio_pl = 0
    symbols_tested = 0

    for s in symbols:
        ticker = s if "." in s else f"{s}.NS"
        print(f"⌛ Testing {ticker}...")
        
        df = yf.download(ticker, period=CONFIG["BACKTEST_PERIOD"], interval="1d", progress=False)
        if df.empty: continue

        # DATA FIX FOR BACKTRADER
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col) for col in df.columns]
        df = df.dropna()

        cerebro = bt.Cerebro()
        data_feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data_feed, name=ticker)
        cerebro.addstrategy(AdvancedRSIStrategy)
        cerebro.broker.setcash(CONFIG["INITIAL_CASH"])
        
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tr")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

        try:
            results = cerebro.run()
            res = results[0].analyzers.tr.get_analysis()
            drawdown = results[0].analyzers.dd.get_analysis()

            if 'total' in res and res.total.total > 0:
                net_profit = cerebro.broker.get_value() - CONFIG["INITIAL_CASH"]
                total_portfolio_pl += net_profit
                symbols_tested += 1
                
                summary = (
                    f"📈 *Result: {ticker}*\n"
                    f"Total Trades: {res.total.total}\n"
                    f"✅ Won: {res.won.total if 'won' in res else 0}\n"
                    f"🔥 Max DD: {drawdown.max.drawdown:.2f}%\n"
                    f"💰 Net P/L: ₹{net_profit:.2f}"
                )
                send_msg(summary)
                print(f"✅ Result sent for {ticker}. Waiting 5s...")
                time.sleep(CONFIG["DELAY_SECONDS"]) # Rate limiting delay
            else:
                print(f"ℹ️ No trades for {ticker}")
                
        except Exception as e:
            print(f"❌ Error testing {ticker}: {e}")

    # FINAL TOTAL UPDATE
    final_message = (
        f"🏁 *COMPLETED ALL BACKTESTS*\n\n"
        f"Total Symbols Tested: {symbols_tested}\n"
        f"💼 *Total Portfolio P/L:* ₹{total_portfolio_pl:.2f}\n"
        f"Initial Capital per stock: ₹{CONFIG['INITIAL_CASH']}"
    )
    send_msg(final_message)
    print("🚀 All tasks complete. Final report sent.")
