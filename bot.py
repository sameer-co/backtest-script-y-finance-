import backtrader as bt
import yfinance as yf
import pandas as pd
import os
import requests
import time

# --- CUSTOMIZABLE PARAMETERS (Modify these as needed) ---
CONFIG = {
    "TOKEN": "8050135427:AAFNQYFpU8lMQ-reJlvLnPYFKc8pyPrHblE",
    "CHAT_ID": "1950462171",
    "INITIAL_CASH": 50000,
    "RISK_PERCENT": 0.02,     # 2% Risk per trade
    "RSI_PERIOD": 40,
    "WMA_PERIOD": 15,
    "ATR_PERIOD": 14,
    "TARGET_RR": 2.5,        # 2.5x Risk-Reward for 50% exit
    "ATR_TRAIL_MULT": 2.0,   # 2.0x ATR for trailing
    "BACKTEST_PERIOD": "2y",
    "WATCHLIST_FILE": "backtest_list.txt"
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
        # 1. ENTRY LOGIC
        if not self.position:
            if self.rsi[0] > self.wma_rsi[0] and self.rsi[-1] <= self.wma_rsi[-1]:
                entry = self.data.close[0]
                yest_low = self.data.low[-1]
                risk_per_share = entry - yest_low

                if risk_per_share > 0:
                    # Risk Management: 2% of Current Portfolio
                    risk_amount = self.broker.get_value() * CONFIG["RISK_PERCENT"]
                    qty = int(risk_amount / risk_per_share)
                    
                    # Cash Check
                    max_qty = int(self.broker.get_cash() / entry)
                    final_qty = min(qty, max_qty)

                    if final_qty > 0:
                        self.buy(size=final_qty)
                        self.stop_loss = yest_low
                        self.target = entry + (risk_per_share * self.p.rr)
                        self.half_booked = False

        # 2. EXIT & TRAILING LOGIC
        elif self.position:
            # Check for Target Hit (50% Exit)
            if not self.half_booked and self.data.high[0] >= self.target:
                self.sell(size=int(self.position.size / 2))
                self.half_booked = True
                # Move Stop to Breakeven once target hit (optional, but safer)
                self.stop_loss = max(self.stop_loss, self.data.open[0])

            # Trailing Stop: ACTIVATES ONLY AFTER 50% IS BOOKED
            if self.half_booked:
                current_atr_trail = self.data.close[0] - (self.atr[0] * self.p.atr_m)
                # Only move SL up, never down
                self.stop_loss = max(self.stop_loss, current_atr_trail)

            # Check for Stop Loss Hit
            if self.data.low[0] <= self.stop_loss:
                self.close()

# --- UTILS ---
def send_msg(text):
    url = f"https://api.telegram.org/bot{CONFIG['TOKEN']}/sendMessage"
    payload = {"chat_id": CONFIG["CHAT_ID"], "text": text, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=10)
    except: pass

# --- SAMPLE WATCHLIST CREATOR ---
def create_sample_watchlist():
    if not os.path.exists(CONFIG["WATCHLIST_FILE"]):
        # Top NSE symbols for backtesting
        samples = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS"]
        with open(CONFIG["WATCHLIST_FILE"], "w") as f:
            f.write("\n".join(samples))
        print(f"Created sample watchlist: {CONFIG['WATCHLIST_FILE']}")

# --- MAIN RUNNER ---
if __name__ == "__main__":
    import requests # Imported here for scope
    create_sample_watchlist()
    
    with open(CONFIG["WATCHLIST_FILE"], "r") as f:
        symbols = [s.strip().upper() for s in f.read().splitlines() if s.strip()]

    for s in symbols:
        ticker = s if "." in s else f"{s}.NS"
        print(f"⌛ Testing {ticker}...")
        
        df = yf.download(ticker, period=CONFIG["BACKTEST_PERIOD"], interval="1d", progress=False)
        if df.empty: continue

        cerebro = bt.Cerebro()
        cerebro.adddata(bt.feeds.PandasData(dataname=df), name=ticker)
        cerebro.addstrategy(AdvancedRSIStrategy)
        cerebro.broker.setcash(CONFIG["INITIAL_CASH"])
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tr")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

        results = cerebro.run()
        res = results[0].analyzers.tr.get_analysis()
        drawdown = results[0].analyzers.dd.get_analysis()

        if 'total' in res and res.total.total > 0:
            total = res.total.total
            won = res.won.total if 'won' in res else 0
            lost = res.lost.total if 'lost' in res else 0
            net_profit = cerebro.broker.get_value() - CONFIG["INITIAL_CASH"]
            
            summary = (
                f"📈 *Result: {ticker}*\n"
                f"Total Trades: {total}\n"
                f"✅ Wins: {won} | ❌ Loss: {lost}\n"
                f"🔥 Max Drawdown: {drawdown.max.drawdown:.2f}%\n"
                f"💰 Net P/L: ₹{net_profit:.2f}\n"
                f"Account: ₹{cerebro.broker.get_value():.2f}"
            )
            send_msg(summary)
            time.sleep(1)
