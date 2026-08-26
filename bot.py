"""
NIFTY 50 T+2 OPEN EXIT BACKTEST
================================

STRATEGY
--------

Trading day T:
    Select stock
    BUY at CLOSE

Trading day T+1:
    HOLD entire trading day

Trading day T+2:
    SELL at OPEN
    Position is now closed

Trading day T+2:
    BUY a new position at CLOSE

Then repeat.

So:

    T Close      -> BUY
    T+1 Entire   -> HOLD
    T+2 Open     -> SELL
    T+2 Close    -> NEW BUY

Only ONE position can exist at a time.

STOCK SELECTION
---------------
1. Take the eligible universe.
2. Calculate 20-day momentum using only information
   available at the current day's close.
3. Rank stocks.
4. Keep the top 20.
5. Buy the #1 ranked stock.

DATA
----
Uses yfinance daily OHLC data.

IMPORTANT:
yfinance does not provide historical NIFTY 50 membership.

If nifty50_history.csv exists, the script uses it.

Expected format:

symbol,start_date,end_date
RELIANCE.NS,2016-01-01,2099-12-31
INFY.NS,2016-01-01,2099-12-31

Otherwise the script uses a fallback basket of 20 large
NSE stocks. That can introduce survivorship bias.

INSTALL
-------
pip install yfinance pandas numpy matplotlib

RUN
---
python nifty_overnight_backtest.py
"""

import os
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INITIAL_CAPITAL = 100000.0

START_DATE = "2016-01-01"
END_DATE = "2026-08-25"

# Number of stocks considered after ranking.
TOP_BASKET_SIZE = 20

# Momentum used for ranking.
MOMENTUM_LOOKBACK = 20

# Historical NIFTY membership file.
NIFTY_HISTORY_FILE = "nifty50_history.csv"

# Output directory.
OUTPUT_DIR = Path("backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# TRANSACTION COST CONFIGURATION
# ============================================================

# Approximate Indian equity delivery costs.
# Adjust these if you want to match your broker exactly.

BROKERAGE_BUY = 0.0
BROKERAGE_SELL = 0.0

# STT
STT_BUY = 0.001
STT_SELL = 0.001

# Approximate exchange transaction charge
EXCHANGE_TXN_RATE = 0.0000297

# SEBI turnover charge
SEBI_RATE = 0.000001

# Stamp duty on buy
STAMP_DUTY_BUY = 0.00015

# GST
GST_RATE = 0.18

# Set this above zero if you want slippage.
SLIPPAGE_RATE = 0.0


# ============================================================
# FALLBACK 20 STOCK UNIVERSE
# ============================================================

# Used only if historical membership CSV does not exist.

FALLBACK_STOCKS = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "TCS.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "LT.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "HINDUNILVR.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "HCLTECH.NS",
    "M&M.NS",
    "NTPC.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_symbol(symbol):

    symbol = str(symbol).strip().upper()

    if symbol.startswith("NSE:"):
        symbol = symbol.replace("NSE:", "")

    if not symbol.endswith(".NS"):
        symbol += ".NS"

    return symbol


def load_historical_membership():

    path = Path(NIFTY_HISTORY_FILE)

    if not path.exists():
        return None

    df = pd.read_csv(path)

    required_columns = {
        "symbol",
        "start_date",
        "end_date"
    }

    if not required_columns.issubset(df.columns):

        raise ValueError(
            "Historical membership CSV must contain:\n"
            "symbol,start_date,end_date"
        )

    df["symbol"] = df["symbol"].apply(clean_symbol)

    df["start_date"] = pd.to_datetime(
        df["start_date"]
    )

    df["end_date"] = pd.to_datetime(
        df["end_date"]
    )

    return df


def is_member(symbol, date, membership):

    if membership is None:

        return symbol in FALLBACK_STOCKS

    rows = membership[
        membership["symbol"] == symbol
    ]

    if rows.empty:
        return False

    valid = rows[
        (rows["start_date"] <= date)
        &
        (rows["end_date"] >= date)
    ]

    return not valid.empty


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_stock_data(symbols):

    print("\nDownloading historical data...\n")

    data = {}

    end_plus_one = (
        pd.Timestamp(END_DATE)
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    for number, symbol in enumerate(symbols, start=1):

        print(
            f"[{number}/{len(symbols)}] Downloading {symbol}"
        )

        df = None

        for attempt in range(3):

            try:

                df = yf.download(
                    symbol,
                    start=START_DATE,
                    end=end_plus_one,
                    interval="1d",
                    auto_adjust=False,
                    progress=False
                )

                if not df.empty:
                    break

            except Exception as e:

                print(
                    f"Attempt {attempt + 1} failed: {e}"
                )

                time.sleep(1)

        if df is None or df.empty:

            print(
                f"Skipping {symbol}: no data"
            )

            continue

        # Handle MultiIndex returned by yfinance.
        if isinstance(df.columns, pd.MultiIndex):

            try:
                df = df.xs(
                    symbol,
                    axis=1,
                    level=1
                )
            except Exception:
                df.columns = df.columns.get_level_values(0)

        df.columns = [
            str(column).lower().replace(" ", "_")
            for column in df.columns
        ]

        required = ["open", "close"]

        if not all(
            column in df.columns
            for column in required
        ):

            print(
                f"Skipping {symbol}: missing OHLC"
            )

            continue

        df.index = pd.to_datetime(df.index)

        # Remove timezone if present.
        if df.index.tz is not None:

            df.index = df.index.tz_localize(None)

        df = df.sort_index()

        df = df.dropna(
            subset=["open", "close"]
        )

        if len(df) > MOMENTUM_LOOKBACK + 2:

            data[symbol] = df

            print(
                f"  OK - {len(df)} trading days"
            )

        else:

            print(
                f"  Skipped - insufficient history"
            )

    return data


# ============================================================
# MOMENTUM CALCULATION
# ============================================================

def calculate_momentum(
    df,
    date
):

    if date not in df.index:
        return None

    location = df.index.get_loc(date)

    if location < MOMENTUM_LOOKBACK:
        return None

    current_close = float(
        df.iloc[location]["close"]
    )

    previous_close = float(
        df.iloc[
            location - MOMENTUM_LOOKBACK
        ]["close"]
    )

    if previous_close <= 0:
        return None

    momentum = (
        current_close / previous_close
    ) - 1

    return momentum


# ============================================================
# SELECT STOCK
# ============================================================

def select_stock(
    entry_date,
    stock_data,
    membership
):

    rankings = []

    for symbol, df in stock_data.items():

        # Only use stocks that belong to the
        # historical universe on this date.
        if not is_member(
            symbol,
            entry_date,
            membership
        ):
            continue

        momentum = calculate_momentum(
            df,
            entry_date
        )

        if momentum is None:
            continue

        rankings.append(
            {
                "symbol": symbol,
                "momentum": momentum
            }
        )

    if not rankings:

        return None, None

    ranking_df = pd.DataFrame(rankings)

    ranking_df = ranking_df.sort_values(
        "momentum",
        ascending=False
    ).reset_index(drop=True)

    # Keep top 20 candidates.
    top_basket = ranking_df.head(
        TOP_BASKET_SIZE
    )

    # Select highest ranked stock.
    selected_stock = top_basket.iloc[0]["symbol"]

    selected_momentum = float(
        top_basket.iloc[0]["momentum"]
    )

    return selected_stock, selected_momentum


# ============================================================
# PRICE VALIDATION
# ============================================================

def get_price(
    stock_data,
    symbol,
    date,
    column
):

    if symbol not in stock_data:
        return None

    df = stock_data[symbol]

    if date not in df.index:
        return None

    value = df.loc[date, column]

    try:

        value = float(value)

    except Exception:

        return None

    if (
        not np.isfinite(value)
        or value <= 0
    ):

        return None

    return value


# ============================================================
# SLIPPAGE
# ============================================================

def apply_slippage(
    price,
    side
):

    if side == "BUY":

        return price * (
            1 + SLIPPAGE_RATE
        )

    elif side == "SELL":

        return price * (
            1 - SLIPPAGE_RATE
        )

    return price


# ============================================================
# COST CALCULATION
# ============================================================

def calculate_transaction_costs(
    buy_value,
    sell_value
):

    brokerage = (
        buy_value * BROKERAGE_BUY
        +
        sell_value * BROKERAGE_SELL
    )

    stt = (
        buy_value * STT_BUY
        +
        sell_value * STT_SELL
    )

    exchange_charges = (
        buy_value + sell_value
    ) * EXCHANGE_TXN_RATE

    sebi_charges = (
        buy_value + sell_value
    ) * SEBI_RATE

    stamp_duty = (
        buy_value
        * STAMP_DUTY_BUY
    )

    gst = GST_RATE * (
        brokerage
        +
        exchange_charges
        +
        sebi_charges
    )

    total_cost = (
        brokerage
        +
        stt
        +
        exchange_charges
        +
        sebi_charges
        +
        stamp_duty
        +
        gst
    )

    return {
        "brokerage": brokerage,
        "stt": stt,
        "exchange_charges": exchange_charges,
        "sebi_charges": sebi_charges,
        "stamp_duty": stamp_duty,
        "gst": gst,
        "total_cost": total_cost
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    stock_data,
    membership
):

    # Build a master list of all trading dates.
    all_dates = set()

    for df in stock_data.values():

        all_dates.update(
            df.index.tolist()
        )

    trading_dates = sorted(
        date
        for date in all_dates
        if (
            date >= pd.Timestamp(START_DATE)
            and
            date <= pd.Timestamp(END_DATE)
        )
    )

    trading_dates = pd.DatetimeIndex(
        trading_dates
    )

    capital = INITIAL_CAPITAL

    trades = []

    equity_curve = [
        {
            "date": trading_dates[0],
            "equity": capital
        }
    ]

    # IMPORTANT:
    #
    # We manually advance the index.
    #
    # Entry:
    #       T close
    #
    # Exit:
    #       T+2 open
    #
    # Next entry:
    #       T+2 close
    #
    # Therefore after exiting, the next trade starts
    # on the same date at close.
    #
    i = 0

    while i < len(trading_dates) - 2:

        entry_date = trading_dates[i]

        exit_date = trading_dates[i + 2]

        # ----------------------------------------------------
        # 1. SELECT STOCK AT ENTRY DAY CLOSE
        # ----------------------------------------------------

        symbol, momentum = select_stock(
            entry_date,
            stock_data,
            membership
        )

        if symbol is None:

            i += 1

            continue

        # ----------------------------------------------------
        # 2. GET ENTRY PRICE
        # ----------------------------------------------------

        raw_entry_price = get_price(
            stock_data,
            symbol,
            entry_date,
            "close"
        )

        # ----------------------------------------------------
        # 3. GET EXIT PRICE
        # ----------------------------------------------------

        raw_exit_price = get_price(
            stock_data,
            symbol,
            exit_date,
            "open"
        )

        if (
            raw_entry_price is None
            or raw_exit_price is None
        ):

            i += 1

            continue

        entry_price = apply_slippage(
            raw_entry_price,
            "BUY"
        )

        exit_price = apply_slippage(
            raw_exit_price,
            "SELL"
        )

        # ----------------------------------------------------
        # 4. 100% CAPITAL ALLOCATION
        # ----------------------------------------------------

        capital_before = capital

        shares = (
            capital_before
            / entry_price
        )

        buy_value = (
            shares
            * entry_price
        )

        sell_value = (
            shares
            * exit_price
        )

        # ----------------------------------------------------
        # 5. COSTS
        # ----------------------------------------------------

        costs = calculate_transaction_costs(
            buy_value,
            sell_value
        )

        total_cost = costs[
            "total_cost"
        ]

        # ----------------------------------------------------
        # 6. P&L
        # ----------------------------------------------------

        gross_pnl = (
            sell_value
            - buy_value
        )

        net_pnl = (
            gross_pnl
            - total_cost
        )

        gross_return = (
            gross_pnl
            / capital_before
        )

        net_return = (
            net_pnl
            / capital_before
        )

        capital = (
            capital_before
            + net_pnl
        )

        # ----------------------------------------------------
        # 7. SAVE TRADE
        # ----------------------------------------------------

        trade = {
            "entry_date": entry_date,
            "exit_date": exit_date,

            "symbol": symbol,

            "momentum_at_entry": momentum,

            "entry_price": entry_price,
            "exit_price": exit_price,

            "shares": shares,

            "capital_before": capital_before,
            "capital_after": capital,

            "buy_value": buy_value,
            "sell_value": sell_value,

            "gross_pnl": gross_pnl,
            "transaction_cost": total_cost,
            "net_pnl": net_pnl,

            "gross_return": gross_return,
            "net_return": net_return,

            "win": net_pnl > 0
        }

        trades.append(trade)

        equity_curve.append(
            {
                "date": exit_date,
                "equity": capital
            }
        )

        # ----------------------------------------------------
        # CRITICAL PART
        # ----------------------------------------------------
        #
        # We exited at T+2 OPEN.
        #
        # We can enter another position at T+2 CLOSE.
        #
        # Therefore next entry index is:
        #
        # i = i + 2
        #
        # NOT i + 1
        #
        # This prevents overlapping positions.
        # ----------------------------------------------------

        i += 2

    trades_df = pd.DataFrame(trades)

    equity_df = pd.DataFrame(
        equity_curve
    )

    equity_df = (
        equity_df
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last"
        )
        .reset_index(drop=True)
    )

    return trades_df, equity_df


# ============================================================
# MAX DRAWDOWN
# ============================================================

def calculate_max_drawdown(
    equity
):

    running_peak = equity.cummax()

    drawdown = (
        equity
        / running_peak
        - 1
    )

    max_drawdown = drawdown.min()

    max_drawdown_amount = (
        equity
        - running_peak
    ).min()

    return (
        max_drawdown,
        max_drawdown_amount
    )


# ============================================================
# SHARPE
# ============================================================

def calculate_sharpe(
    returns
):

    returns = returns.dropna()

    if len(returns) < 2:
        return np.nan

    std = returns.std()

    if std == 0:
        return np.nan

    return (
        returns.mean()
        / std
        * math.sqrt(252)
    )


# ============================================================
# SORTINO
# ============================================================

def calculate_sortino(
    returns
):

    returns = returns.dropna()

    downside = returns[
        returns < 0
    ]

    if len(downside) == 0:
        return np.nan

    downside_deviation = np.sqrt(
        (downside ** 2).mean()
    )

    if downside_deviation == 0:
        return np.nan

    return (
        returns.mean()
        / downside_deviation
        * math.sqrt(252)
    )


# ============================================================
# CONSECUTIVE WINS / LOSSES
# ============================================================

def consecutive_stats(
    trades
):

    max_wins = 0
    max_losses = 0

    current_wins = 0
    current_losses = 0

    for pnl in trades["net_pnl"]:

        if pnl > 0:

            current_wins += 1
            current_losses = 0

        else:

            current_losses += 1
            current_wins = 0

        max_wins = max(
            max_wins,
            current_wins
        )

        max_losses = max(
            max_losses,
            current_losses
        )

    return (
        max_wins,
        max_losses
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    trades,
    equity
):

    total_trades = len(trades)

    if total_trades == 0:

        return {}

    final_capital = float(
        trades.iloc[-1]["capital_after"]
    )

    total_pnl = (
        final_capital
        - INITIAL_CAPITAL
    )

    total_return = (
        final_capital
        / INITIAL_CAPITAL
        - 1
    )

    winners = trades[
        trades["net_pnl"] > 0
    ]

    losers = trades[
        trades["net_pnl"] < 0
    ]

    win_rate = (
        len(winners)
        / total_trades
    )

    avg_win = (
        winners["net_return"].mean()
        if len(winners) > 0
        else 0
    )

    avg_loss = (
        losers["net_return"].mean()
        if len(losers) > 0
        else 0
    )

    expectancy = (
        win_rate * avg_win
        +
        (1 - win_rate) * avg_loss
    )

    gross_profit = winners[
        "net_pnl"
    ].sum()

    gross_loss = abs(
        losers["net_pnl"].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    # Equity returns
    equity = equity.copy()

    equity["return"] = equity[
        "equity"
    ].pct_change()

    sharpe = calculate_sharpe(
        equity["return"]
    )

    sortino = calculate_sortino(
        equity["return"]
    )

    max_dd, max_dd_amount = (
        calculate_max_drawdown(
            equity["equity"]
        )
    )

    annual_volatility = (
        equity["return"].std()
        * math.sqrt(252)
    )

    # CAGR
    start_date = equity[
        "date"
    ].iloc[0]

    end_date = equity[
        "date"
    ].iloc[-1]

    years = (
        end_date
        - start_date
    ).days / 365.25

    if years > 0:

        cagr = (
            (
                final_capital
                / INITIAL_CAPITAL
            )
            **
            (1 / years)
        ) - 1

    else:

        cagr = np.nan

    max_wins, max_losses = (
        consecutive_stats(trades)
    )

    return {

        "Initial Capital":
            INITIAL_CAPITAL,

        "Final Capital":
            final_capital,

        "Total P&L":
            total_pnl,

        "Total Return":
            total_return,

        "CAGR":
            cagr,

        "Total Trades":
            total_trades,

        "Winning Trades":
            len(winners),

        "Losing Trades":
            len(losers),

        "Win Rate":
            win_rate,

        "Average Trade":
            trades["net_return"].mean(),

        "Average Winner":
            avg_win,

        "Average Loser":
            avg_loss,

        "Expectancy Per Trade":
            expectancy,

        "Profit Factor":
            profit_factor,

        "Max Drawdown":
            max_dd,

        "Max Drawdown Amount":
            max_dd_amount,

        "Sharpe Ratio":
            sharpe,

        "Sortino Ratio":
            sortino,

        "Annualized Volatility":
            annual_volatility,

        "Best Trade":
            trades["net_return"].max(),

        "Worst Trade":
            trades["net_return"].min(),

        "Max Consecutive Wins":
            max_wins,

        "Max Consecutive Losses":
            max_losses,

        "Total Transaction Costs":
            trades[
                "transaction_cost"
            ].sum()
    }


# ============================================================
# DISPLAY REPORT
# ============================================================

def print_metrics(metrics):

    print("\n")
    print("=" * 65)
    print("BACKTEST RESULTS")
    print("=" * 65)

    for key, value in metrics.items():

        if (
            "Return" in key
            or "Rate" in key
            or "CAGR" in key
            or "Drawdown" in key
            or "Winner" in key
            or "Loser" in key
            or "Expectancy" in key
            or "Trade" in key
            and isinstance(value, float)
        ):

            if np.isfinite(value):

                print(
                    f"{key:<30}: "
                    f"{value * 100:.2f}%"
                )

            else:

                print(
                    f"{key:<30}: N/A"
                )

        elif (
            "Capital" in key
            or "P&L" in key
            or "Amount" in key
            or "Costs" in key
        ):

            print(
                f"{key:<30}: "
                f"₹{value:,.2f}"
            )

        elif isinstance(value, float):

            if np.isfinite(value):

                print(
                    f"{key:<30}: "
                    f"{value:.4f}"
                )

            else:

                print(
                    f"{key:<30}: N/A"
                )

        else:

            print(
                f"{key:<30}: {value}"
            )

    print("=" * 65)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    trades,
    equity,
    metrics
):

    trades.to_csv(
        OUTPUT_DIR / "trades.csv",
        index=False
    )

    equity.to_csv(
        OUTPUT_DIR / "equity_curve.csv",
        index=False
    )

    metrics_df = pd.DataFrame(
        list(metrics.items()),
        columns=[
            "Metric",
            "Value"
        ]
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "metrics.csv",
        index=False
    )

    # Equity curve plot.
    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        equity["date"],
        equity["equity"]
    )

    plt.title(
        "Portfolio Equity Curve"
    )

    plt.xlabel("Date")

    plt.ylabel(
        "Portfolio Value (INR)"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "equity_curve.png",
        dpi=150
    )

    plt.close()

    print("\nResults saved to:")

    print(
        OUTPUT_DIR / "trades.csv"
    )

    print(
        OUTPUT_DIR / "equity_curve.csv"
    )

    print(
        OUTPUT_DIR / "metrics.csv"
    )

    print(
        OUTPUT_DIR / "equity_curve.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)

    print(
        "NIFTY T+2 OPEN EXIT BACKTEST"
    )

    print("=" * 65)

    print("\nStrategy rules:")

    print(
        "BUY  : Day T close"
    )

    print(
        "HOLD : Entire Day T+1"
    )

    print(
        "SELL : Day T+2 open"
    )

    print(
        "NEXT BUY : Day T+2 close"
    )

    print(
        "Capital allocation: 100%"
    )

    print(
        f"Momentum lookback: "
        f"{MOMENTUM_LOOKBACK} days"
    )

    # --------------------------------------------------------
    # LOAD MEMBERSHIP
    # --------------------------------------------------------

    membership = load_historical_membership()

    if membership is None:

        print("\nWARNING")

        print(
            "Historical NIFTY membership file "
            "not found."
        )

        print(
            "Using fallback 20-stock basket."
        )

        symbols = FALLBACK_STOCKS

    else:

        symbols = sorted(
            membership[
                "symbol"
            ].unique()
        )

        print(
            f"\nHistorical universe loaded: "
            f"{len(symbols)} symbols"
        )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    stock_data = download_stock_data(
        symbols
    )

    print(
        f"\nUsable stocks: "
        f"{len(stock_data)}"
    )

    if not stock_data:

        raise RuntimeError(
            "No stock data downloaded."
        )

    # --------------------------------------------------------
    # RUN BACKTEST
    # --------------------------------------------------------

    trades, equity = run_backtest(
        stock_data,
        membership
    )

    if trades.empty:

        raise RuntimeError(
            "No trades generated."
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        trades,
        equity
    )

    print_metrics(
        metrics
    )

    # --------------------------------------------------------
    # SAMPLE TRADES
    # --------------------------------------------------------

    print("\nFIRST 10 TRADES\n")

    columns = [
        "entry_date",
        "exit_date",
        "symbol",
        "entry_price",
        "exit_price",
        "net_pnl",
        "net_return",
        "capital_after"
    ]

    print(
        trades[
            columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(
        trades,
        equity,
        metrics
    )


if __name__ == "__main__":

    main()
