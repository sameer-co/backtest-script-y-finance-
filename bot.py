import backtrader as bt
import yfinance as yf
import pandas as pd
import os
import requests
import time
import collections
import numpy as np

# Compatibility patch for Python 3.10+
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable

CONFIG = {
    "TOKEN":          "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg",
    "CHAT_ID":        "1950462171",
    "INITIAL_CASH":   50000,
    "RISK_PERCENT":   0.02,
    "BACKTEST_PERIOD": "2y",
    "WATCHLIST_FILE": "watchlist.txt",
    "TEST_TIMEFRAMES": ["2h", "4h" ,"1d"],
    "COMMISSION":     0.001,   # 0.1% round-trip (brokerage + fees)
    "DELAY_SECONDS":  5,
}

STATS = {
    "global":     {"profit": 0, "trades": 0, "wins": 0, "gross_p": 0, "gross_l": 0, "max_dd": 0},
    "timeframes": {tf: {"profit": 0, "trades": 0, "wins": 0} for tf in CONFIG["TEST_TIMEFRAMES"]},
    "best":       {"name": "", "profit": -999999},
}


class AdvancedRSIStrategy(bt.Strategy):
    params = (
        ('rsi_p',    40),
        ('wma_p',    15),
        ('atr_p',    14),
        ('rr',       2.5),
        ('atr_m',    2.0),
        ('adx_p',    14),
        ('adx_limit', 25),
        ('sma_p',    15),   # FIX Warn #2: was 7 (noise) → 15 (meaningful short-term trend)
    )

    def __init__(self):
        self.rsi     = bt.indicators.RSI(self.data.close, period=self.p.rsi_p)
        self.wma_rsi = bt.indicators.WMA(self.rsi,        period=self.p.wma_p)
        self.atr     = bt.indicators.ATR(self.data,       period=self.p.atr_p)
        self.adx     = bt.indicators.ADX(self.data,       period=self.p.adx_p)
        self.sma     = bt.indicators.SMA(self.data.close, period=self.p.sma_p)

        # ATR moving average for volatility gate (ATR-gated RSI idea)
        self.atr_ma  = bt.indicators.SMA(self.atr,        period=20)

        # Trade state — reset properly via notify_trade (FIX Bug #3)
        self.stop_loss   = None
        self.target      = None
        self.half_booked = False
        self.entry_price = None

    # FIX Bug #3: reset state cleanly after every closed trade
    def notify_trade(self, trade):
        if trade.isclosed:
            self.stop_loss   = None
            self.target      = None
            self.half_booked = False
            self.entry_price = None

    def next(self):
        if not self.position:
            rsi_cross    = self.rsi[0] > self.wma_rsi[0] and self.rsi[-1] <= self.wma_rsi[-1]
            trend_strong = self.adx[0] >= self.p.adx_limit
            above_sma    = self.data.close[0] > self.sma[0]
            # ATR gate: only enter when volatility is expanding (ATR > its own MA)
            vol_expanding = self.atr[0] > self.atr_ma[0]

            if rsi_cross and trend_strong and above_sma and vol_expanding:
                entry = self.data.close[0]
                risk  = entry - self.data.low[-1]

                if risk > 0:
                    qty       = int((self.broker.get_value() * CONFIG["RISK_PERCENT"]) / risk)
                    final_qty = min(qty, int(self.broker.get_cash() / entry))

                    if final_qty > 0:
                        self.buy(size=final_qty)
                        self.entry_price = entry
                        self.stop_loss   = self.data.low[-1]
                        self.target      = entry + (risk * self.p.rr)
                        self.half_booked = False

        elif self.position:
            # Guard: stop_loss should never be None inside a position
            if self.stop_loss is None:
                self.stop_loss = self.data.low[-1]

            # 1. Partial profit at RR target
            if not self.half_booked and self.data.high[0] >= self.target:
                # FIX Bug #2: guard against size=0 on small positions
                half_size = max(1, int(self.position.size / 2))
                self.sell(size=half_size)
                self.half_booked = True
                # Move stop to breakeven (entry price, not open[0])
                self.stop_loss = max(self.stop_loss, self.entry_price)

            # 2. ATR trailing stop (active after half-booking)
            if self.half_booked:
                trail = self.data.close[0] - (self.atr[0] * self.p.atr_m)
                self.stop_loss = max(self.stop_loss, trail)

            # 3. Hard stop exit
            if self.data.low[0] <= self.stop_loss:
                self.close()


# ── HELPERS ──────────────────────────────────────────────────────────────────

def send_msg(text: str) -> None:
    url = f"https://api.telegram.org/bot{CONFIG['TOKEN']}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": CONFIG["CHAT_ID"], "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception:
        pass


def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "1h":
        return df
    tf_map = {"1d": "D", "4h": "4h", "2h": "2h"}
    rule = tf_map.get(timeframe, timeframe)
    return df.resample(rule).apply(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    ).dropna()


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(CONFIG["WATCHLIST_FILE"]):
        print(f"Error: {CONFIG['WATCHLIST_FILE']} not found.")
    else:
        with open(CONFIG["WATCHLIST_FILE"]) as f:
            symbols = [s.strip().upper() for s in f.read().splitlines() if s.strip()]

        for s in symbols:
            ticker = s if "." in s else f"{s}.NS"
            raw_df = yf.download(
                ticker,
                period=CONFIG["BACKTEST_PERIOD"],
                interval="1h",
                progress=False,
            )

            if raw_df.empty:
                continue

            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)
            raw_df.columns = [str(col) for col in raw_df.columns]

            for tf in CONFIG["TEST_TIMEFRAMES"]:
                print(f"Analyzing {ticker} @ {tf}...")
                df = resample_data(raw_df.copy(), tf)

                cerebro = bt.Cerebro()
                cerebro.adddata(bt.feeds.PandasData(dataname=df), name=f"{ticker}_{tf}")
                cerebro.addstrategy(AdvancedRSIStrategy)
                cerebro.broker.setcash(CONFIG["INITIAL_CASH"])

                # FIX Warn #3: add realistic commission
                cerebro.broker.setcommission(commission=CONFIG["COMMISSION"])

                # FIX Bug #1: fill at current bar close so entry price matches close[0]
                cerebro.broker.set_coc(True)

                cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tr")
                cerebro.addanalyzer(bt.analyzers.DrawDown,       _name="dd")
                cerebro.addanalyzer(bt.analyzers.SharpeRatio,    _name="sr",
                                    riskfreerate=0.065, annualize=True)

                try:
                    results = cerebro.run()
                    res = results[0].analyzers.tr.get_analysis()
                    dd  = results[0].analyzers.dd.get_analysis()
                    sr  = results[0].analyzers.sr.get_analysis()

                    if 'total' in res and res.total.total > 0:
                        pnl        = cerebro.broker.get_value() - CONFIG["INITIAL_CASH"]
                        n_trades   = res.total.total
                        n_wins     = res.won.total  if 'won'  in res else 0
                        gross_p    = res.won.pnl.total  if 'won'  in res else 0
                        gross_l    = abs(res.lost.pnl.total) if 'lost' in res else 0
                        max_dd_val = dd.max.drawdown
                        sharpe     = sr.get('sharperatio', 0) or 0

                        # Global totals (FIX Warn #4: labelled correctly as "sum across runs")
                        STATS["global"]["profit"]  += pnl
                        STATS["global"]["trades"]  += n_trades
                        STATS["global"]["wins"]    += n_wins
                        STATS["global"]["gross_p"] += gross_p
                        STATS["global"]["gross_l"] += gross_l
                        STATS["global"]["max_dd"]   = max(STATS["global"]["max_dd"], max_dd_val)

                        STATS["timeframes"][tf]["profit"] += pnl
                        STATS["timeframes"][tf]["trades"] += n_trades
                        STATS["timeframes"][tf]["wins"]   += n_wins

                        if pnl > STATS["best"]["profit"]:
                            STATS["best"] = {"name": f"{ticker} ({tf})", "profit": pnl}

                        print(
                            f"  {ticker} {tf}: ₹{pnl:.0f} | "
                            f"trades={n_trades} | acc={n_wins/n_trades*100:.1f}% | "
                            f"sharpe={sharpe:.2f} | dd={max_dd_val:.1f}%"
                        )

                except Exception as e:
                    print(f"  Error {ticker} {tf}: {e}")

            time.sleep(1)

        # ── Final report ──────────────────────────────────────────────────────
        g        = STATS["global"]
        win_rate = (g["wins"]   / g["trades"] * 100) if g["trades"] > 0 else 0
        pf       = (g["gross_p"] / g["gross_l"])      if g["gross_l"] > 0 else 0
        exp      = (g["profit"] / g["trades"])         if g["trades"] > 0 else 0

        tf_report = ""
        for tf, data in STATS["timeframes"].items():
            acc = (data["wins"] / data["trades"] * 100) if data["trades"] > 0 else 0
            tf_report += f"• *{tf}:* ₹{data['profit']:.0f}  |  Trades: {data['trades']}  |  Acc: {acc:.1f}%\n"

        master_report = (
            f"🏛 *BACKTEST PERFORMANCE REPORT*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ _P/L below = sum across all independent symbol runs_\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 *TOTAL P/L (all runs):* ₹{g['profit']:.2f}\n"
            f"📊 *OVERALL ACCURACY:*  {win_rate:.2f}%\n"
            f"⚖️ *PROFIT FACTOR:*     {pf:.2f}\n"
            f"📉 *WORST DRAWDOWN:*    {g['max_dd']:.2f}%\n"
            f"🎯 *EXPECTANCY:*        ₹{exp:.2f} / trade\n"
            f"🔢 *TOTAL TRADES:*      {g['trades']}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⏳ *TIMEFRAME BREAKDOWN*\n"
            f"{tf_report}"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🌟 *STAR PERFORMER:* {STATS['best']['name']} "
            f"(₹{STATS['best']['profit']:.0f})\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📅 *PERIOD:* {CONFIG['BACKTEST_PERIOD']}  "
            f"|  *COMMISSION:* {CONFIG['COMMISSION']*100:.1f}%\n"
            f"🏁 *Report finalised.*"
        )
        send_msg(master_report)
        print("\n" + master_report)
