"""
Main driver.
"""

import argparse
import logging.config

# from rich.logging import RichHandler
import datetime as dt
import numpy as np
import pandas as pd
from dateutil.tz import tz
import sys
from typing import List
import utils.equity_crypto_reader as equity_crypto_reader
import utils.io_ as io
import visualization.graphing as graph
from pathlib import Path
import yaml
from logging_config import log_config_yaml

config = io.load_config_file(Path.cwd().parent / "config-batten.yaml")
logging.config.dictConfig(yaml.load(log_config_yaml, Loader=yaml.UnsafeLoader))
exec_log = logging.getLogger("batten.expose")
file_away = logging.getLogger("batten.write")

session_id = io.session_id()
timed_subdir = dt.datetime.now(tz=tz.tzutc()).isoformat()
now = dt.datetime.now()
localtz = tz.gettz().tzname(now)


def get_options(argv: List[str]) -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    # Add arguments to the parser object
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--t0", action="store")
    parser.add_argument("--t1", action="store")
    return parser.parse_args(argv)


def btc_joy(
    btc: pd.DataFrame,
    floats: List[str] = ["open", "high", "low", "close", "volume"],
    date_field: str = 'date',
) -> pd.DataFrame:
    """
    Pull most up-to-date or load pre-existing price data_tasks, reindex, cast dates to datetime,
    strip numerals from column names, ensure continuous variables are of dtype `float`.
    Parameters
    ----------
    btc
    floats
    date_field

    Returns
    -------

    """
    btc.columns = btc.columns.str.replace(r"^\d+\.\s*", "", regex=True)
    btc[floats] = btc[floats].astype(float)
    btc[date_field] = pd.to_datetime(btc[date_field], format="%Y-%m-%d")
    return btc


def moving_averages(
    btc: pd.DataFrame,
    windows: List[int] = [7, 14, 30, 200],
) -> pd.DataFrame:

    for wind in windows:
        # right-aligned moving averages (consider centered)
        btc.loc[:, "ma" + str(wind)] = (
            btc["close"].rolling(window=wind).mean()
        )  # .ewm(com=0.5).mean()
        # Breach to upside
        btc.loc[:, "ma" + str(wind) + "b"] = np.where(
            btc.close > btc["ma" + str(wind)], 1, 0
        )
        # Breach to downside
        btc.loc[:, "ma" + str(wind) + "b_"] = np.where(
            btc.close < btc["ma" + str(wind)], 1, 0
        )
    return btc


if __name__ == "__main__":
    options = get_options(sys.argv[1:])

    exec_log.info(f"{now.strftime('%Y-%m-%d %H:%M:%S ') + localtz} | {session_id}")

    extra_parameters = {"observation_start": options.t0, "observation_end": options.t1}

    # `path_data` contains full path through the file stem
    # `serialization_format` manages the file extension
    btc_json = io.contingent_load(
        path_data=(Path.cwd().parent / config["paths"]["data_tasks"]["interim"]) / "btc.json",
        entity="crypto",
        ticker="BTC",
        function="DIGITAL_CURRENCY_DAILY",
        market="USD",
        refresh=options.refresh,
        serialize_and_save=True,
        serialization_format="json",
        unique_file=None,
    )
    if list(btc_json.keys())[0] == "Error Message":
        raise Exception(f"Data retrieval failed")
    btc = equity_crypto_reader.security_json_to_dataframe(
        btc_json, key_="Time Series (Digital Currency Daily)", rename_={"index": "date"}
    )
    exec_log.info(f"Cleaning")
    btc = btc_joy(btc)
    exec_log.info(f"Augment: Price changes")
    btc = (
        btc.assign(dp1=btc.close - btc.open)
        .assign(dp2=btc.high - btc.low)
        .assign(dp3=btc.close.shift() - btc.close)
        .assign(dp3p=btc.close.shift() / btc.close - 1)
    )
    exec_log.info(f"Augment: Moving averages and breaches")
    btc = moving_averages(btc)
    exec_log.debug(btc.head())

    # Next: locate first (downward) breach as a (sell) buy signal

    # Locate peak-to-troughs, and peak to `delta` points

    # Construct a P&L from the prices at each buy and sell signal

    if options.plot:
        btc_chart = graph.altair_ts_line(
            btc,
            "index",
            "close",
            _title="Bitcoin",
            y_title="Closing price (USD)",
            x_title=f"Days since {btc.date.min()}",
            tooltip_fld="close",
        )

    log_dir = Path.cwd().parent / config["paths"]["reports"]["logs"]
    completed_log = (log_dir / "work.log").rename(
        log_dir / (timed_subdir + "_work" + ".log")
    )


