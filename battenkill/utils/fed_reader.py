"""Operate APIs that retrieve economic data_tasks from the Federal Reserve and
Bank for International Settlements.

https://github.com/gw-moore/pyfredapi
Fed funds rates across the term structure
Expectations of same: Fed funds futures

"""

import pyfredapi as pf
from typing import List
import pandas as pd


# Open webpage for an economic series
# gdp_info.open_url()

def fred_single(series_id: str,
                **extra_parameters) -> pd.DataFrame:
    """Retrieve single series from the federal reserve."""

    data = pf.get_series(series_id, **extra_parameters)
    return data


def fred_collection(series_ids: List[str], **extra_parameters) -> pf.SeriesCollection:
    """Retrieve collection of series"""

    collection_ = pf.SeriesCollection(series_id=series_ids, **extra_parameters)
    return collection_
