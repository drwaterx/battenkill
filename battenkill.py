"""
Main driver.
"""
# built-in
import argparse
import datetime as dt
from dateutil.tz import tz
import logging.config
from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union
import yaml

# third party
import dotenv
import numpy as np
import pandas as pd
# from rich.logging import RichHandler

# first party
import battenkill.utils.fed_reader as fred
import battenkill.utils.equity_crypto_reader as equity_crypto_reader
import battenkill.utils.io_ as io
import battenkill.visualization.graphing as graph
from battenkill.logging_config import log_config_yaml

# Load environment variables (API keys, etc.)
dotenv.load_dotenv()

# Load configuration file and set up logging
config = io.load_config_file(Path.cwd() / "config-batten.yaml")
logging.config.dictConfig(yaml.load(log_config_yaml, Loader=yaml.FullLoader))
exlog = logging.getLogger("batten.expose")
file_away = logging.getLogger("batten.write")

# Define session metadata
session_id = io.session_id()
timed_subdir = dt.datetime.now(tz=tz.tzutc()).isoformat()
now = dt.datetime.now()
localtz = tz.gettz().tzname(now)


def get_options(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t0",
        action="store",
        default=None,
        help="Start time for data collection (YYYY-MM-DD)",
    )
    parser.add_argument("--t1", action="store")
    return parser.parse_args(argv)


def fed_funds_repo_pull_and_chart(
    plot_data: bool = True,
    save_chart_: bool = False,
    dash_: bool = True,
):
    """
    Load, collect, and plot Fed and repo rates.

    Parameters
    ----------
    plot_data
    save_chart_
    dash_

    Returns
    -------

    """
    effr = fred.fred_single(series_id="FEDFUNDS", **extra_parameters)
    sofr = fred.fred_single(series_id="SOFR", **extra_parameters)
    sc = fred.fred_collection(["DFEDTARL", "DFEDTARU"], **extra_parameters)
    ffrt = sc["DFEDTARL"].df.merge(sc["DFEDTARU"].df)

    data = [
        effr,
        sofr,
        ffrt,
    ]

    if plot_data:
        chart = graph.compound_fed_repo_rates(data, save_chart=save_chart_)
    if plot_data and dash_:
        try:
            chart.show()
        except TypeError as e:
            exlog.error(f"Chart generation failed: {e}")
    return chart


def bitcoin_pull_and_chart(
    cols=None,
    refresh_: bool = False,
):
    if cols is None:
        cols = ['open', 'high', 'low', 'close', 'volume']
    btc_json = io.contingent_load(
        path_data=(Path.cwd().parent / "data_tasks/interim/btc.json"),
        entity="crypto",
        ticker="BTC",
        function="DIGITAL_CURRENCY_DAILY",
        market="USD",
        refresh=refresh_,
        serialize_and_save=True,
        serialization_format="json",
        unique_file=None,
    )
    btc = equity_crypto_reader.security_json_to_dataframe(
        btc_json, key_="Time Series (Digital Currency Daily)", rename_={"index": "date"}
    )
    btc.date = pd.to_datetime(btc.date, format="%Y-%m-%d")
    btc.columns = btc.columns.str.replace(r'^\d+\.\s*', '', regex=True)
    btc[cols] = btc[cols].astype(float)
    btc = (
        btc
        .assign(dp1=btc.close - btc.open)
        .assign(dp2=btc.high - btc.low)
        .assign(dp3=btc.close.shift() - btc.close)
        .assign(dp3p=btc.close.shift() / btc.close - 1)
    )

    exlog.debug(btc.head(10))

    btc_chart = graph.altair_ts_scatter(
        btc,
        "date",
        "close",
        _title="Bitcoin",
        y_title="Closing price (USD)",
        x_title="Date",
        tooltip_fld=["date", "close"],
        save=True,
        chart_file_stem=f"btc_{btc.date.min()}-to-present",
    )
    btc_chart.show()


def main(

):
    options = get_options(sys.argv[1:])
    exlog.info(f"{now.strftime('%Y-%m-%d %H:%M:%S ') + localtz} | {session_id}")
    extra_parameters = {"observation_start": options.t0, "observation_end": options.t1}

    # fed_funds_repo_pull_and_chart()
    bitcoin_pull_and_chart()

    log_dir = Path.cwd() / config["paths"]["reports"]["logs"]
    completed_log = (log_dir / "work.log").rename(
        log_dir / (timed_subdir + "_work" + ".log")
    )


if __name__ == "__main__":
    main()
