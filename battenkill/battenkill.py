"""
Main driver.
"""

import argparse
import logging.config

# from rich.logging import RichHandler
import datetime as dt

import pandas as pd
from dateutil.tz import tz
import sys
from typing import List
import utils.fed_reader as fred
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
    parser.add_argument("--econ", action="store")  # dash denotes an optional argument
    parser.add_argument("--t0", action="store")
    parser.add_argument("--t1", action="store")
    return parser.parse_args(argv)


if __name__ == "__main__":
    options = get_options(sys.argv[1:])

    exec_log.info(f"{now.strftime('%Y-%m-%d %H:%M:%S ') + localtz} | {session_id}")

    extra_parameters = {"observation_start": options.t0, "observation_end": options.t1}

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
                exec_log.error(f"Chart generation failed: {e}")
        return chart

    # fed_funds_repo_pull_and_chart()

    # `path_data` contains full path through the file stem
    # `serialization_format` manages the file extension
    btc_json = io.contingent_load(
        path_data=(Path.cwd().parent / config["paths"]["data"]["interim"]) / "btc.json",
        entity="crypto",
        ticker="BTC",
        function="DIGITAL_CURRENCY_DAILY",
        market="CNY",
        refresh=False,
        serialize_and_save=True,
        serialization_format="json",
        unique_file=None,
    )
    btc = equity_crypto_reader.security_json_to_dataframe(
        btc_json, key_="Time Series (Digital Currency Daily)", rename_={"index": "date"}
    )
    btc.date = pd.to_datetime(btc.date)

    exec_log.debug(btc.head())

    btc_chart = graph.altair_ts_scatter(
        btc,
        "date",
        "4b. close (USD)",
        _title="Bitcoin",
        y_title="Closing price (USD)",
        tooltip_fld="4b. close (USD)",
    )
    btc_chart.show()

    log_dir = Path.cwd().parent / config["paths"]["reports"]["logs"]
    completed_log = (log_dir / "work.log").rename(
        log_dir / (timed_subdir + "_work" + ".log")
    )
