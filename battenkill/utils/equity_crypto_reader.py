""" Operate APIs that retrieve equity, ETF, and cryptocurrency and token prices.
AAPL
AMZN
CRWD
FDIS
SNOW
UPST
M
TSLA

IJS i-Shares small cap value ETF - long
IVW large cap growth - short
DTH high dividend yield European ETF
"""
import os
import requests
from typing import Optional

import pandas as pd


def security_prices_alphavantage(
    ticker: str,
    function: str = "TIME_SERIES_DAILY",
    market: str = "",
    *,
    outputsize: str = "compact",
) -> dict:
    """
    Obtains security prices from alphavantage.

    Parameters
    ----------
    market
    ticker
    function
        Determines which API endpoint from which to obtain the data_tasks. For a cryptocurrency,
        use 'DIGITAL_CURRENCY_DAILY'.
    outputsize

    Returns
    -------

    """
    # Construct the URL for Alpha Vantage data_tasks pull
    url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&market={market}&outputsize={outputsize}&apikey={os.getenv('AlphaVantage_API_KEY')}"
    # url =  "https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=EUR&apikey=demo"
    r = requests.get(url)
    return r.json()  # dict


def security_json_to_dataframe(
    data: dict,
    key_: str = "Time Series (Daily)",
    convert_to_datetime: str = True,
    rename_: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Converts price time series within a nested dictionary containing from an
    AlphaVantage API endpoint to a dataframe.  Does not retain metadata.

    DataFrame is indexed by date in ISO format; columns: open, high, low, close, volume

    Parameters
    ----------
    rename_
    convert_to_datetime
    key_
    data
        Dictionary containing nested data_tasks.

    Returns
    -------
        DataFrame of security price time series
    """
    data_df = pd.DataFrame.from_dict(data[key_], orient="index")

    if convert_to_datetime:
        data_df.index = pd.to_datetime(data_df.index)
    if rename_:
        data_df = data_df.reset_index().rename(columns=rename_)
    return data_df
