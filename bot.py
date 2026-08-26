"""
NIFTY-50 T+2 OPEN EXIT / CLOSE ENTRY BACKTEST
================================================

ONE-SYMBOL-AT-A-TIME DATA ARCHITECTURE

This script:

1. Gets a list of NIFTY-50 symbols.
2. Downloads each symbol separately from yfinance.
3. Saves each symbol locally in data/cache/.
4. On future runs, loads cached data instead of downloading again.
5. Builds a daily cross-sectional ranking.
6. Selects the highest-ranked stock from the top-20 basket.
7. Uses 100% of capital in ONE stock.
8. Buys at T CLOSE.
9. Holds through T+1.
10. Sells at T+2 OPEN.
11. At T+2 CLOSE, enters the next position.
12. Repeats.

TRADE TIMELINE
--------------

            T              T+1             T+2
            │               │               │
        BUY CLOSE          HOLD          SELL OPEN
                                            │
                                            │
                                      position closed
                                            │
                                        BUY CLOSE
                                            │
                                            ▼
                                      next position


INSTALL
-------

pip install yfinance pandas numpy matplotlib


RUN
---

python nifty_overnight_backtest.py


OPTIONAL ENVIRONMENT VARIABLES
------------------------------

START_DATE=2016-01-01
END_DATE=2026-08-26
INITIAL_CAPITAL=100000
TOP_BASKET_SIZE=20
MOMENTUM_LOOKBACK=20


IMPORTANT
---------

yfinance provides historical price data.

yfinance does NOT provide a reliable historical NIFTY-50
constituent history.

Therefore this script has two modes:

MODE 1:
-------
If "nifty50_history.csv" exists:

symbol,start_date,end_date

RELIANCE.NS,2016-01-01,2099-12-31
...

then historical membership is respected.

MODE 2:
-------
If the file does not exist, the script uses the built-in
NIFTY basket below.

This means MODE 2 can contain survivorship bias.

For a rigorous historical NIFTY-50 study, use historical
membership data eventually.


"""

from __future__ import annotations

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

INITIAL_CAPITAL = float(
    os.getenv(
        "INITIAL_CAPITAL",
        "100000"
    )
)

START_DATE = os.getenv(
    "START_DATE",
    "2016-01-01"
)

END_DATE = os.getenv(
    "END_DATE",
    "2026-08-26"
)

TOP_BASKET_SIZE = int(
    os.getenv(
        "TOP_BASKET_SIZE",
        "20"
    )
)

MOMENTUM_LOOKBACK = int(
    os.getenv(
        "MOMENTUM_LOOKBACK",
        "20"
    )
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "cache"
)

RESULTS_DIR = (
    BASE_DIR
    / "backtest_results"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRANSACTION COSTS
# ============================================================

BROKERAGE_BUY = 0.0
BROKERAGE_SELL = 0.0

STT_BUY = 0.001
STT_SELL = 0.001

EXCHANGE_TXN_RATE = 0.0000297

SEBI_RATE = 0.000001

STAMP_DUTY_BUY = 0.00015

GST_RATE = 0.18

SLIPPAGE_RATE = 0.0


# ============================================================
# FALLBACK NIFTY UNIVERSE
# ============================================================

NIFTY_50_SYMBOLS = [

    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "ETERNAL.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "ITC.NS",
    "JIOFIN.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHRIRAMFIN.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "TRENT.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
]


# ============================================================
# SYMBOL NORMALIZATION
# ============================================================

def clean_symbol(symbol):

    symbol = str(
        symbol
    ).strip().upper()

    if symbol.startswith(
        "NSE:"
    ):

        symbol = symbol[
            4:
        ]

    if not symbol.endswith(
        ".NS"
    ):

        symbol += ".NS"

    return symbol


# ============================================================
# HISTORICAL MEMBERSHIP
# ============================================================

def load_historical_membership():

    path = (
        BASE_DIR
        / "nifty50_history.csv"
    )

    if not path.exists():

        print(
            "\nHistorical membership file "
            "not found."
        )

        print(
            "Using built-in NIFTY-50 universe."
        )

        print(
            "NOTE: this can contain "
            "survivorship bias."
        )

        return None

    df = pd.read_csv(
        path
    )

    required = {
        "symbol",
        "start_date",
        "end_date"
    }

    if not required.issubset(
        df.columns
    ):

        raise ValueError(
            "\n"
            "nifty50_history.csv must contain:\n"
            "symbol,start_date,end_date"
        )

    df["symbol"] = (
        df["symbol"]
        .apply(clean_symbol)
    )

    df["start_date"] = (
        pd.to_datetime(
            df["start_date"]
        )
    )

    df["end_date"] = (
        pd.to_datetime(
            df["end_date"]
        )
    )

    return df


# ============================================================
# GET SYMBOL LIST
# ============================================================

def get_symbols(
    membership
):

    if membership is None:

        return sorted(
            set(
                NIFTY_50_SYMBOLS
            )
        )

    return sorted(
        membership[
            "symbol"
        ]
        .dropna()
        .unique()
        .tolist()
    )


# ============================================================
# HISTORICAL MEMBERSHIP CHECK
# ============================================================

def is_member(
    symbol,
    date,
    membership
):

    if membership is None:

        return (
            symbol
            in NIFTY_50_SYMBOLS
        )

    rows = membership[
        membership[
            "symbol"
        ]
        == symbol
    ]

    if rows.empty:

        return False

    valid = rows[
        (
            rows[
                "start_date"
            ]
            <= date
        )
        &
        (
            rows[
                "end_date"
            ]
            >= date
        )
    ]

    return not valid.empty


# ============================================================
# CACHE PATH
# ============================================================

def cache_path(
    symbol
):

    safe_symbol = (
        symbol
        .replace(
            ".NS",
            ""
        )
        .replace(
            "/",
            "_"
        )
    )

    return (
        DATA_DIR
        / f"{safe_symbol}.csv"
    )


# ============================================================
# DOWNLOAD ONE SYMBOL
# ============================================================

def download_one_symbol(
    symbol
):

    path = cache_path(
        symbol
    )

    # --------------------------------------------------------
    # CACHE EXISTS
    # --------------------------------------------------------

    if path.exists():

        print(
            f"  CACHE: {symbol}"
        )

        df = pd.read_csv(
            path,
            index_col=0,
            parse_dates=True
        )

        df.index = pd.to_datetime(
            df.index
        )

        return df


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    print(
        f"  DOWNLOAD: {symbol}"
    )

    end_plus_one = (
        pd.Timestamp(
            END_DATE
        )
        + pd.Timedelta(
            days=1
        )
    ).strftime(
        "%Y-%m-%d"
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
                progress=False,
                threads=False
            )

            if (
                df is not None
                and not df.empty
            ):

                break

        except Exception as e:

            print(
                f"    retry "
                f"{attempt + 1}: "
                f"{e}"
            )

            time.sleep(
                2
            )

    if (
        df is None
        or df.empty
    ):

        print(
            f"    FAILED: {symbol}"
        )

        return None


    # --------------------------------------------------------
    # FIX YFINANCE MULTIINDEX
    # --------------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        try:

            df = df.xs(
                symbol,
                axis=1,
                level=1
            )

        except Exception:

            df.columns = (
                df.columns
                .get_level_values(
                    0
                )
            )


    # --------------------------------------------------------
    # NORMALIZE COLUMN NAMES
    # --------------------------------------------------------

    df.columns = [

        str(column)
        .strip()
        .lower()
        .replace(
            " ",
            "_"
        )

        for column
        in df.columns

    ]


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    required = {
        "open",
        "close"
    }

    if not required.issubset(
        df.columns
    ):

        print(
            f"    FAILED: "
            f"{symbol} "
            f"missing Open/Close"
        )

        return None


    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    df.index = pd.to_datetime(
        df.index
    )

    if df.index.tz is not None:

        df.index = (
            df.index
            .tz_localize(None)
        )


    # --------------------------------------------------------
    # NUMERIC DATA
    # --------------------------------------------------------

    for column in [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume"
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df = (
        df
        .sort_index()
        .dropna(
            subset=[
                "open",
                "close"
            ]
        )
    )


    # --------------------------------------------------------
    # SAVE CACHE
    # --------------------------------------------------------

    df.to_csv(
        path
    )

    print(
        f"    SAVED: {path}"
    )

    return df


# ============================================================
# DOWNLOAD ALL SYMBOLS
# ONE SYMBOL AT A TIME
# ============================================================

def load_all_data(
    symbols
):

    data = {}

    print(
        "\n"
        + "=" * 65
    )

    print(
        "FETCHING HISTORICAL DATA"
    )

    print(
        "=" * 65
    )

    for number, symbol in enumerate(
        symbols,
        start=1
    ):

        print(
            f"\n[{number}/{len(symbols)}]"
        )

        df = download_one_symbol(
            symbol
        )

        if df is None:

            continue

        if len(df) <= (
            MOMENTUM_LOOKBACK
            + 2
        ):

            print(
                "    Insufficient history"
            )

            continue

        data[
            symbol
        ] = df

    print(
        "\nSuccessfully loaded:"
        f" {len(data)} symbols"
    )

    return data


# ============================================================
# BUILD MARKET TRADING CALENDAR
# ============================================================

def build_trading_calendar(
    data
):

    """
    We need a common market calendar.

    Instead of using the union of all stock dates,
    use the intersection of available dates from the
    largest/most complete reference stock.

    For NIFTY stocks, RELIANCE is normally a good
    reference. If unavailable, use the stock with
    the largest number of observations.
    """

    if not data:

        raise RuntimeError(
            "No data available."
        )

    if "RELIANCE.NS" in data:

        reference_symbol = (
            "RELIANCE.NS"
        )

    else:

        reference_symbol = max(
            data,
            key=lambda symbol:
                len(data[symbol])
        )

    calendar = data[
        reference_symbol
    ].index

    calendar = calendar[
        (
            calendar
            >= pd.Timestamp(
                START_DATE
            )
        )
        &
        (
            calendar
            <= pd.Timestamp(
                END_DATE
            )
        )
    ]

    return pd.DatetimeIndex(
        calendar
    )


# ============================================================
# MOMENTUM
# ============================================================

def calculate_momentum(
    df,
    date
):

    if date not in df.index:

        return None

    position = (
        df.index
        .get_loc(date)
    )

    if position < (
        MOMENTUM_LOOKBACK
    ):

        return None

    current_close = float(
        df.iloc[
            position
        ]["close"]
    )

    previous_close = float(
        df.iloc[
            position
            - MOMENTUM_LOOKBACK
        ]["close"]
    )

    if (
        current_close <= 0
        or previous_close <= 0
    ):

        return None

    return (
        current_close
        /
        previous_close
        - 1
    )


# ============================================================
# SELECT STOCK
# ============================================================

def select_stock(
    entry_date,
    data,
    membership
):

    rankings = []

    for symbol, df in data.items():

        # Historical membership
        if not is_member(
            symbol,
            entry_date,
            membership
        ):

            continue

        momentum = (
            calculate_momentum(
                df,
                entry_date
            )
        )

        if momentum is None:

            continue

        rankings.append(
            {
                "symbol":
                    symbol,

                "momentum":
                    momentum
            }
        )

    if not rankings:

        return None, None

    ranking = pd.DataFrame(
        rankings
    )

    ranking = ranking.sort_values(
        [
            "momentum",
            "symbol"
        ],
        ascending=[
            False,
            True
        ]
    ).reset_index(
        drop=True
    )

    ranking["rank"] = (
        np.arange(
            1,
            len(ranking) + 1
        )
    )

    top20 = ranking.head(
        TOP_BASKET_SIZE
    )

    if top20.empty:

        return None, ranking

    selected = (
        top20.iloc[0]["symbol"]
    )

    selected_score = float(
        top20.iloc[0][
            "momentum"
        ]
    )

    return (
        selected,
        selected_score
    )


# ============================================================
# GET PRICE
# ============================================================

def get_price(
    data,
    symbol,
    date,
    column
):

    if symbol not in data:

        return None

    df = data[
        symbol
    ]

    if date not in df.index:

        return None

    try:

        value = float(
            df.loc[
                date,
                column
            ]
        )

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

def execute_price(
    price,
    side
):

    if side == "BUY":

        return (
            price
            * (
                1
                + SLIPPAGE_RATE
            )
        )

    return (
        price
        * (
            1
            - SLIPPAGE_RATE
        )
    )


# ============================================================
# TRANSACTION COSTS
# ============================================================

def calculate_costs(
    buy_value,
    sell_value
):

    brokerage = (
        buy_value
        * BROKERAGE_BUY
        +
        sell_value
        * BROKERAGE_SELL
    )

    stt = (
        buy_value
        * STT_BUY
        +
        sell_value
        * STT_SELL
    )

    exchange = (
        buy_value
        +
        sell_value
    ) * EXCHANGE_TXN_RATE

    sebi = (
        buy_value
        +
        sell_value
    ) * SEBI_RATE

    stamp_duty = (
        buy_value
        * STAMP_DUTY_BUY
    )

    gst = (
        brokerage
        +
        exchange
        +
        sebi
    ) * GST_RATE

    total = (
        brokerage
        +
        stt
        +
        exchange
        +
        sebi
        +
        stamp_duty
        +
        gst
    )

    return {
        "brokerage":
            brokerage,

        "stt":
            stt,

        "exchange":
            exchange,

        "sebi":
            sebi,

        "stamp_duty":
            stamp_duty,

        "gst":
            gst,

        "total":
            total
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    data,
    membership,
    calendar
):

    capital = (
        INITIAL_CAPITAL
    )

    trades = []

    equity = []

    i = 0

    while (
        i
        <
        len(calendar) - 2
    ):

        # ----------------------------------------------------
        # ENTRY DATE
        # ----------------------------------------------------

        entry_date = (
            calendar[i]
        )

        # ----------------------------------------------------
        # EXIT DATE
        # ----------------------------------------------------

        exit_date = (
            calendar[i + 2]
        )


        # ----------------------------------------------------
        # SELECT STOCK
        # ----------------------------------------------------

        symbol, momentum = (
            select_stock(
                entry_date,
                data,
                membership
            )
        )

        if symbol is None:

            i += 1

            continue


        # ----------------------------------------------------
        # ENTRY PRICE
        # ----------------------------------------------------

        raw_entry = get_price(
            data,
            symbol,
            entry_date,
            "close"
        )


        # ----------------------------------------------------
        # EXIT PRICE
        # ----------------------------------------------------

        raw_exit = get_price(
            data,
            symbol,
            exit_date,
            "open"
        )


        if (
            raw_entry is None
            or raw_exit is None
        ):

            i += 1

            continue


        entry_price = (
            execute_price(
                raw_entry,
                "BUY"
            )
        )

        exit_price = (
            execute_price(
                raw_exit,
                "SELL"
            )
        )


        # ----------------------------------------------------
        # 100% CAPITAL
        # ----------------------------------------------------

        capital_before = (
            capital
        )

        shares = (
            capital_before
            /
            entry_price
        )

        buy_value = (
            shares
            *
            entry_price
        )

        sell_value = (
            shares
            *
            exit_price
        )


        # ----------------------------------------------------
        # COST
        # ----------------------------------------------------

        costs = (
            calculate_costs(
                buy_value,
                sell_value
            )
        )

        total_cost = (
            costs["total"]
        )


        # ----------------------------------------------------
        # P&L
        # ----------------------------------------------------

        gross_pnl = (
            sell_value
            -
            buy_value
        )

        net_pnl = (
            gross_pnl
            -
            total_cost
        )

        net_return = (
            net_pnl
            /
            capital_before
        )


        # ----------------------------------------------------
        # UPDATE CAPITAL
        # ----------------------------------------------------

        capital_after = (
            capital_before
            +
            net_pnl
        )

        capital = (
            capital_after
        )


        # ----------------------------------------------------
        # TRADE RECORD
        # ----------------------------------------------------

        trades.append({

            "entry_date":
                entry_date,

            "exit_date":
                exit_date,

            "symbol":
                symbol,

            "momentum":
                momentum,

            "entry_price":
                entry_price,

            "exit_price":
                exit_price,

            "shares":
                shares,

            "capital_before":
                capital_before,

            "buy_value":
                buy_value,

            "sell_value":
                sell_value,

            "gross_pnl":
                gross_pnl,

            "transaction_cost":
                total_cost,

            "net_pnl":
                net_pnl,

            "net_return":
                net_return,

            "capital_after":
                capital_after,

            "win":
                net_pnl > 0

        })


        # ----------------------------------------------------
        # EQUITY
        # ----------------------------------------------------

        equity.append({

            "date":
                exit_date,

            "equity":
                capital

        })


        # ----------------------------------------------------
        # VERY IMPORTANT
        # ----------------------------------------------------
        #
        # Current trade:
        #
        # i       = T
        # i + 1   = T+1
        # i + 2   = T+2 EXIT
        #
        # New position is allowed at:
        #
        # T+2 CLOSE
        #
        # Therefore:
        #
        # next entry = i + 2
        #
        # This guarantees:
        #
        # NO OVERLAPPING POSITIONS
        #
        # ----------------------------------------------------

        i += 2


    trades = pd.DataFrame(
        trades
    )

    equity = pd.DataFrame(
        equity
    )

    return (
        trades,
        equity
    )


# ============================================================
# MAX DRAWDOWN
# ============================================================

def max_drawdown(
    equity
):

    peak = (
        equity
        .cummax()
    )

    drawdown = (
        equity
        /
        peak
        - 1
    )

    max_dd = (
        drawdown.min()
    )

    amount = (
        equity
        -
        peak
    ).min()

    return (
        max_dd,
        amount
    )


# ============================================================
# SHARPE
# ============================================================

def sharpe_ratio(
    returns
):

    returns = (
        returns
        .dropna()
    )

    if len(returns) < 2:

        return np.nan

    std = (
        returns.std()
    )

    if std == 0:

        return np.nan

    return (
        returns.mean()
        /
        std
        *
        math.sqrt(252)
    )


# ============================================================
# SORTINO
# ============================================================

def sortino_ratio(
    returns
):

    returns = (
        returns
        .dropna()
    )

    downside = (
        returns[
            returns < 0
        ]
    )

    if downside.empty:

        return np.nan

    downside_dev = np.sqrt(
        (
            downside ** 2
        ).mean()
    )

    if downside_dev == 0:

        return np.nan

    return (
        returns.mean()
        /
        downside_dev
        *
        math.sqrt(252)
    )


# ============================================================
# CONSECUTIVE WINS / LOSSES
# ============================================================

def streaks(
    trades
):

    max_wins = 0
    max_losses = 0

    wins = 0
    losses = 0

    for pnl in (
        trades[
            "net_pnl"
        ]
    ):

        if pnl > 0:

            wins += 1
            losses = 0

        else:

            losses += 1
            wins = 0

        max_wins = max(
            max_wins,
            wins
        )

        max_losses = max(
            max_losses,
            losses
        )

    return (
        max_wins,
        max_losses
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    trades,
    equity
):

    if trades.empty:

        return {}

    final_capital = float(
        trades.iloc[-1][
            "capital_after"
        ]
    )

    total_pnl = (
        final_capital
        -
        INITIAL_CAPITAL
    )

    total_return = (
        final_capital
        /
        INITIAL_CAPITAL
        -
        1
    )

    winners = trades[
        trades[
            "net_pnl"
        ] > 0
    ]

    losers = trades[
        trades[
            "net_pnl"
        ] < 0
    ]

    total_trades = (
        len(trades)
    )

    win_rate = (
        len(winners)
        /
        total_trades
    )

    avg_win = (
        winners[
            "net_return"
        ].mean()
        if not winners.empty
        else 0
    )

    avg_loss = (
        losers[
            "net_return"
        ].mean()
        if not losers.empty
        else 0
    )

    expectancy = (
        win_rate
        *
        avg_win
        +
        (
            1
            -
            win_rate
        )
        *
        avg_loss
    )

    gross_profit = (
        winners[
            "net_pnl"
        ].sum()
    )

    gross_loss = abs(
        losers[
            "net_pnl"
        ].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            /
            gross_loss
        )

    else:

        profit_factor = np.inf


    # --------------------------------------------------------
    # EQUITY RETURNS
    # --------------------------------------------------------

    equity = (
        equity
        .sort_values(
            "date"
        )
        .drop_duplicates(
            "date",
            keep="last"
        )
        .copy()
    )

    equity[
        "daily_return"
    ] = (
        equity[
            "equity"
        ]
        .pct_change()
    )


    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    max_dd, dd_amount = (
        max_drawdown(
            equity[
                "equity"
            ]
        )
    )


    # --------------------------------------------------------
    # SHARPE / SORTINO
    # --------------------------------------------------------

    sharpe = (
        sharpe_ratio(
            equity[
                "daily_return"
            ]
        )
    )

    sortino = (
        sortino_ratio(
            equity[
                "daily_return"
            ]
        )
    )


    # --------------------------------------------------------
    # CAGR
    # --------------------------------------------------------

    start = pd.to_datetime(
        equity[
            "date"
        ].iloc[0]
    )

    end = pd.to_datetime(
        equity[
            "date"
        ].iloc[-1]
    )

    years = (
        end - start
    ).days / 365.25

    if years > 0:

        cagr = (
            (
                final_capital
                /
                INITIAL_CAPITAL
            )
            **
            (
                1
                /
                years
            )
            -
            1
        )

    else:

        cagr = np.nan


    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    volatility = (
        equity[
            "daily_return"
        ].std()
        *
        math.sqrt(252)
    )


    # --------------------------------------------------------
    # STREAKS
    # --------------------------------------------------------

    max_wins, max_losses = (
        streaks(
            trades
        )
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    return {

        "initial_capital":
            INITIAL_CAPITAL,

        "final_capital":
            final_capital,

        "total_pnl":
            total_pnl,

        "total_return":
            total_return,

        "cagr":
            cagr,

        "total_trades":
            total_trades,

        "winning_trades":
            len(winners),

        "losing_trades":
            len(losers),

        "win_rate":
            win_rate,

        "average_trade":
            trades[
                "net_return"
            ].mean(),

        "average_winner":
            avg_win,

        "average_loser":
            avg_loss,

        "expectancy":
            expectancy,

        "profit_factor":
            profit_factor,

        "max_drawdown":
            max_dd,

        "max_drawdown_amount":
            dd_amount,

        "annual_volatility":
            volatility,

        "sharpe":
            sharpe,

        "sortino":
            sortino,

        "best_trade":
            trades[
                "net_return"
            ].max(),

        "worst_trade":
            trades[
                "net_return"
            ].min(),

        "max_consecutive_wins":
            max_wins,

        "max_consecutive_losses":
            max_losses,

        "total_transaction_cost":
            trades[
                "transaction_cost"
            ].sum()

    }


# ============================================================
# REPORT
# ============================================================

def print_report(
    metrics
):

    print(
        "\n"
        + "=" * 65
    )

    print(
        "FINAL BACKTEST REPORT"
    )

    print(
        "=" * 65
    )

    money_metrics = {
        "initial_capital",
        "final_capital",
        "total_pnl",
        "max_drawdown_amount",
        "total_transaction_cost"
    }

    percent_metrics = {
        "total_return",
        "cagr",
        "win_rate",
        "average_trade",
        "average_winner",
        "average_loser",
        "expectancy",
        "max_drawdown",
        "annual_volatility",
        "best_trade",
        "worst_trade"
    }

    for key, value in metrics.items():

        label = (
            key
            .replace(
                "_",
                " "
            )
            .title()
        )

        if key in money_metrics:

            print(
                f"{label:<30}"
                f": ₹{value:,.2f}"
            )

        elif key in percent_metrics:

            if np.isfinite(value):

                print(
                    f"{label:<30}"
                    f": {value * 100:.2f}%"
                )

            else:

                print(
                    f"{label:<30}"
                    ": N/A"
                )

        elif isinstance(
            value,
            float
        ):

            if np.isfinite(value):

                print(
                    f"{label:<30}"
                    f": {value:.4f}"
                )

            else:

                print(
                    f"{label:<30}"
                    ": N/A"
                )

        else:

            print(
                f"{label:<30}"
                f": {value}"
            )

    print(
        "=" * 65
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    trades,
    equity,
    metrics
):

    trades_path = (
        RESULTS_DIR
        /
        "trades.csv"
    )

    equity_path = (
        RESULTS_DIR
        /
        "equity_curve.csv"
    )

    metrics_path = (
        RESULTS_DIR
        /
        "metrics.csv"
    )

    plot_path = (
        RESULTS_DIR
        /
        "equity_curve.png"
    )

    trades.to_csv(
        trades_path,
        index=False
    )

    equity.to_csv(
        equity_path,
        index=False
    )

    pd.DataFrame(
        list(
            metrics.items()
        ),
        columns=[
            "metric",
            "value"
        ]
    ).to_csv(
        metrics_path,
        index=False
    )


    # --------------------------------------------------------
    # EQUITY CURVE
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        equity[
            "date"
        ],
        equity[
            "equity"
        ]
    )

    plt.title(
        "NIFTY-50 T+2 Strategy Equity Curve"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Portfolio Value ₹"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=150
    )

    plt.close()


    print(
        "\nResults:"
    )

    print(
        trades_path
    )

    print(
        equity_path
    )

    print(
        metrics_path
    )

    print(
        plot_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 65
    )

    print(
        "NIFTY-50 T+2 BACKTEST"
    )

    print(
        "=" * 65
    )

    print(
        f"Period:"
        f" {START_DATE}"
        f" -> "
        f"{END_DATE}"
    )

    print(
        f"Initial capital:"
        f" ₹{INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Top basket:"
        f" {TOP_BASKET_SIZE}"
    )

    print(
        f"Momentum:"
        f" {MOMENTUM_LOOKBACK} days"
    )


    # --------------------------------------------------------
    # MEMBERSHIP
    # --------------------------------------------------------

    membership = (
        load_historical_membership()
    )


    # --------------------------------------------------------
    # SYMBOLS
    # --------------------------------------------------------

    symbols = get_symbols(
        membership
    )

    print(
        f"\nSymbols to process:"
        f" {len(symbols)}"
    )


    # --------------------------------------------------------
    # ONE SYMBOL AT A TIME
    # --------------------------------------------------------

    data = load_all_data(
        symbols
    )


    if not data:

        raise RuntimeError(
            "No historical data available."
        )


    # --------------------------------------------------------
    # MARKET CALENDAR
    # --------------------------------------------------------

    calendar = (
        build_trading_calendar(
            data
        )
    )

    print(
        f"\nTrading days:"
        f" {len(calendar)}"
    )


    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------

    trades, equity = (
        run_backtest(
            data,
            membership,
            calendar
        )
    )


    if trades.empty:

        raise RuntimeError(
            "No trades generated."
        )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = (
        calculate_metrics(
            trades,
            equity
        )
    )


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print_report(
        metrics
    )


    # --------------------------------------------------------
    # SAMPLE TRADES
    # --------------------------------------------------------

    print(
        "\nFirst 10 trades:"
    )

    print(
        trades[
            [
                "entry_date",
                "exit_date",
                "symbol",
                "entry_price",
                "exit_price",
                "net_pnl",
                "net_return",
                "capital_after"
            ]
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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
