from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional
from altair import Chart
import yaml
import json
import logging
import datetime as dt
import pandas as pd
import numpy
import pyarrow
import uuid
import battenkill.utils.equity_crypto_reader as equity_crypto_reader

from battenkill.utils.quiver import (
    arrow_to_parquet,
    pandas_to_arrow,
    arrow_to_feather,
    load_parquet,
    load_feather,
)

exec_log = logging.getLogger("batten.expose")


def session_id():
    """Provides a unique identifier based on the host ID and current
    time as a 32-character lowercase hexadecimal string.
    """
    return uuid.uuid1().hex


def new_dir(path: Path, prnts: bool = True, exist: bool = True) -> None:
    """Creates new directory, if it does not already exist.

    Parameters
    ----------
    path
        Desired path
    prnts
        Whether to include parent directories, relative to the current working directory
    exist
        Whether to ignore `FileExistsError` exceptions
    """
    if not path.exists():
        path.mkdir(parents=prnts, exist_ok=exist)


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Loads a configuration mapping object with contents
    of a given file.
    :param config_path: Path to be read.
    :returns: mapping with configuration parameter values
    """
    with config_path.open() as config_file:
        document = yaml.load(config_file, Loader=yaml.UnsafeLoader)  # SafeLoader
    return document


def path_to_data(subdirectory: str, file: str):
    path_file = Path.cwd() / subdirectory / file  # Path('.').resolve().parent
    try:
        path_file.exists()
        return path_file
    except FileNotFoundError as ex:
        exec_log.info(f"Path invalid; directory and/or file is incorrect: {ex}")


def load_csv(
    subdirectory: str,
    file: str,
    categories: List[str] = None,
    na_filter: bool = True,  # replaces empty strings with nan
    parse_dt: bool = True,
    infer_dt_format: bool = True,
    set_dtype=False,
):
    full_path = path_to_data(subdirectory, file)
    if set_dtype:
        dtype_dict = {}
        for key in categories:
            dtype_dict[key] = "category"
    else:
        dtype_dict = None
    data = pd.read_csv(
        full_path,
        parse_dates=parse_dt,
        infer_datetime_format=infer_dt_format,
        dtype=dtype_dict,
    )
    exec_log.info(
        f"Loaded {len(data):,} records with {len(data.columns)} fields from {full_path.as_posix()}."
    )
    return data


def save_csv(
    data: pd.DataFrame,
    path: Path,
    prefix: str = "_",
    what: str = "",
    entity: str = "",
    subject: str = "",
    _index: bool = False,
    idx_lab: Union[None, str] = None,
) -> None:
    """Serializes tabular data_tasks as comma-separated text.

    :param data: tabular data_tasks to save
    :param path: path to the directory in which to save the data_tasks
    :param prefix: first characters in the file name
    :param what: for logging, a description of what is being saved
    :param entity: the category of the subject whose data_tasks are being saved
    :param subject: the specific entity whose data_tasks are being saved
    :returns: Nothing (file saved)
    """
    if not path.exists():
        new_dir(path)
    file_stem = prefix + subject
    try:
        data.to_csv(
            (path / file_stem).with_suffix(".csv"), index=_index, index_label=idx_lab
        )
    except:
        if type(data) == numpy.ndarray:
            numpy.savetxt((path / file_stem).with_suffix(".csv"), data, delimiter=",")
        else:
            print(f"Cannot save data_tasks of type {type(data)}.")
    exec_log.info(f"Saved {what} of {entity} {subject} to {path}.")


def save_excel(datasets: dict, path: Union[Path, str], file: str) -> None:
    """
    Writes one or more pandas dataframes to an Excel workbook,
    one sheet per dataframe. See pandas.ExcelWriter doc for
    additional options.

    Parameters
    ----------
    datasets
        Dictionary of pandas DataFrames
    path
        Directories and filename for the Excel workbook.
    """
    if not path.exists():
        new_dir(path)
        # If the folder was absent, the file needed by ExcelWriter is also absent; create it
        pd.DataFrame().to_excel((path / file).with_suffix(".xlsx"))
    else:
        # As necessary, provide ExcelWriter with a pre-existing file
        if not (path / file).with_suffix(".xlsx").exists():
            pd.DataFrame().to_excel((path / file).with_suffix(".xlsx"))
    with pd.ExcelWriter(
        (path / file).with_suffix(".xlsx"),
        mode="w",  # append
        engine="openpyxl",
        if_sheet_exists="replace",
        date_format="YYYY-MM-DD",
        datetime_format="YYYY-MM-DD HH:MM:SS",
    ) as writer:
        for key in datasets.keys():
            (
                datasets[key]
                .reset_index()
                .rename(columns={"index": "txn_idx"})
                .to_excel(writer, sheet_name=key)
            )
            exec_log.info(f"Saved {key} to Excel: {path}.")


def save_altair(
        charts: Union[Chart, Tuple[Chart, ...]],
        cpath: Path,
        chart_file_stem: str,
        format_: str = '.png',
        image_scalar: int = 2,
) -> None:
    """Writes json of one or more (superimposed) altair chart(s) to an HTML file.

    Parameters
    ----------
    image_scalar
        Helps ensure images are not grainy.
    format_
        Typically png or html
    charts
        An altair chart or sum of multiple charts. If multiple charts, they
        must be enclosed in parentheses.
    cpath
        Directory to which to write the file.
    chart_file_stem
        Completes the file path
    """
    if not cpath.exists():
        new_dir(cpath)
    charts.save(
        (cpath / chart_file_stem).with_suffix(format_).as_posix(),
        scale_factor=image_scalar
    )


def default_date(object: Any) -> Union[Any, Dict[str, Any]]:
    """Provide a function to be used by the `json` module during serialization.
    JSON data_tasks documents can only have dict, list, str, int, float, bool, and None types.
    Datetimes are common but cannot be serialized; hence, convert to a string that can be
    easily converted back after deserialization.
    """
    if isinstance(object, dt.datetime):
        return {"$date": object.isoformat()}
    return object


def as_date(object: Dict[str, Any]) -> Union[Any, Dict[str, Any]]:
    """During JSON deserialization, checks each object that's decoded to see if
    it has a single field named `$date`. If so, replace with a `datetime` object.
    """
    if {"$date"} == set(object.keys()):
        return dt.datetime.fromisoformat(object["$date"])
    return object


def to_json(
    data: Union[dict, pd.DataFrame, pd.Series, numpy.ndarray],
    path: Path,
    prefix: str = "",
    what: str = "",
    entity: str = "",
    subject: str = "",
) -> None:
    """
    Serializes tabular data_tasks to JavaScript Object Notation.
    When the `json` module can't serialize an object, the module
    passes the object to the given `default` function.

    Parameters
    ----------
    data
        The data_tasks to be serialized.
    path
        Directory to which to store the serialized data_tasks.
    prefix
        Leading characters of the file name.
    what
        Log message: description of what is being serialized.
    entity
        General name of the subject whose data_tasks are being serialized.
    subject
        Specific name of the subject whose data_tasks are being serialize.
    """
    if len((path/prefix).suffix) == 0:
        full_path = (path/prefix).with_suffix(".json")
    else:
        full_path = path/prefix  # (prefix + subject)

    if not full_path.parent.exists():
        new_dir(full_path.parent)

    # given function scope, unclear if necessary to avoid changing input data_tasks
    # data_tasks = data_tasks.copy()

    if isinstance(data, pd.DataFrame) | isinstance(data, pd.Series):
        data = data.to_json(
            orient="records",
            date_format="iso",
            date_unit="s",
            index=True,
        )
    elif isinstance(data, numpy.ndarray):
        data = data.tolist()
        exec_log.warning(
            f"Serialized keyless array; field names may need to be repopulated."
        )
    with full_path.open("w") as target_file:
        json.dump(
            data,
            target_file,
            default=default_date,
        )
    exec_log.info(f"Saved {what} of {entity} {subject} to {path}.")


def dict_from_json(
    path: Path,
    filename: str = '',
) -> dict:
    """ Read JSON string from a file and convert to a dictionary.
    """
    with (path / filename).open() as source_file:
        document = json.load(source_file, object_hook=as_date)
    return document


def df_from_json(
    path: Path,
    filename: str = '',
) -> pd.DataFrame:
    """Read JSON string from a file and convert to a DataFrame.
    Intermediate `json.load` provides datetime conversion prior
    to `pd.read_json`, as the latter converts datetime strings
    under a narrower set of conditions.
    """
    return pd.DataFrame.from_dict(dict_from_json(path, filename), orient='index')


def df_to_markdown_file(
        data: pd.DataFrame,
        path: Path,
        prefix: str = "_",
        what: Optional[str] = None,
        entity: str = "",
        subject: str = "",
) -> None:
    """Converts pandas DataFrame to a markdown table and writes it to a text file.
    """
    o = path/(prefix + '.md')
    o.write_text(data.to_markdown())
    exec_log.info(f"Saved {what} of {entity} {subject} to {o}")


def df_to_latex(
       data: pd.DataFrame,
        path: Path,
        prefix: str = "_",
        float_format_: str = "%.3f",
        caption: Union[str, Tuple]='',
        what: Optional[str] = None,
        entity: str = "",
        subject: str = "",
) -> None:
    """Converts pandas DataFrame to a LaTex longtable and writes it to a tex file.

    Parameters
    ----------
    caption
        If a tuple, it will output a short and full caption.
    """
    o = path/(prefix + '.tex')
    data.to_latex(
        o,
        float_format=float_format_,
        longtable=True,
        caption=caption,
        label=what,
    )
    exec_log.info(f"Saved {what} of {entity} {subject} to {o}")


def contingent_load(
    path_data: Path,
    entity: str,
    ticker: str,
    function: str,
    market: str,
    *,
    index_: Optional[List[str]] = None,
    timestamps: Optional[List[str]] = None,
    refresh: bool = False,
    deserialize_to: str = "pandas",
    serialize_and_save: bool = True,
    serialization_format: str = "json",
    unique_file: Optional[str] = None,
) -> Union[pd.DataFrame, pyarrow.Table]:
    """

    Parameters
    ----------
    market
    function
    ticker
    entity
        The category of entity for which data_tasks are being requested; an equity,
        fixed income, crypto asset, or economic series.  Consider using a TypedDict
    path_data
        Path, including filename, of either a pre-existing file or the file
        in which to serialize the data_tasks after querying the remote repository.
    index_
        Fields with which to index the DataFrame.
    timestamps
        Fields to convert to datetimes.
    refresh
        Whether to query the remote repository regardless of whether the data_tasks file exists.
    deserialize_to
        Once the file is found, determines whether data_tasks is deserialized to a
        `pandas.DataFrame` or an `Arrow.Table`.
    serialize_and_save
        Whether to (over)write the new or refreshed data_tasks to disk.
    serialization_format
        Format in which to serialize the data_tasks; options include Apache Arrow's
        feather format or CSV. Could replace with logic on `full_path` suffix.
    unique_file
        A string that uniquely names a file by being appended to `file_data`;
        if `None`, the necessary information is obtained from `file_data`.

    Returns
    -------
        Pandas DtaFrame or Arrow Table containing the data_tasks queried from
        a market data_tasks provider or deserialized from a file.
    """

    # Define the path in which to look for a data_tasks file;
    # should it include a unique identifier?
    # Also assign a path for a new/refreshed data_tasks.
    if unique_file:
        full_path = (path_data.parent / (path_data.stem + unique_file)).with_suffix(
            "." + serialization_format
        )
    else:
        full_path = path_data.with_suffix("." + serialization_format)

    exec_log.debug(f"{full_path} | exists: {full_path.exists()}")

    if full_path.exists() & ~refresh:
        assert (
            path_data.suffix == "." + serialization_format
        ), "file serialization arguments are inconsistent."
        exec_log.info(f"Deserializing data_tasks from {full_path}")
        if full_path.suffix == ".json":
            data = dict_from_json(full_path)
            # exec_log.warning(f"Deserialization of json file is tbd.")
        elif full_path.suffix == ".parquet":
            data = load_parquet(full_path)
        elif full_path.suffix == ".feather":
            data = load_feather(
                full_path,
                destination=deserialize_to,
            )
        elif full_path.suffix == ".csv":
            data = load_csv(
                full_path,
                index_col=index_,
                parse_dt=timestamps,
            )
    else:
        # call market data_tasks API
        if entity == "equity":
            data = equity_crypto_reader.security_prices_alphavantage(
                ticker,
                function,
                market,
            )
        elif entity == "crypto":
            data = equity_crypto_reader.security_prices_alphavantage(
                ticker,
                function,
                market,
            )
        # elif entity == 'fixed income':
        #     fed_reader._(
        #         ticker,
        #         function,
        #         market,
        #     )
        # elif entity == 'econ':
        #     equity_crypto_reader.security_prices_alphavantage(
        #     )
        else:
            exec_log.warning(f"{entity} is not a valid security type.")
            return
        if serialize_and_save:
            exec_log.info(f"{type(data)} data_tasks received | to be serialized as {serialization_format}.")
            if serialization_format == "json":
                if isinstance(data, dict):
                    to_json(data, full_path)
            elif serialization_format == "parquet":
                if isinstance(data, pyarrow.Table):
                    arrow_to_parquet(data, path=full_path)
                elif isinstance(data, pd.DataFrame):
                    arrow_to_parquet(pandas_to_arrow(data), path=full_path)
                else:
                    exec_log.warning(
                        f"Serialization of data_tasks type {type(data)} to parquet is not supported."
                    )
            elif serialization_format == "feather":
                if isinstance(data, dict):
                    exec_log.warning(f"Serialization of data_tasks type {type(data)} to feather is tbd.")
                    # likely need to construct a line-delimited json string first
                    # nested_data = [
                    #     {'category': 'A', 'values': [10, 20, 30]},
                    #     {'category': 'B', 'values': [15, 25, 35]}
                    # ]
                    # # Convert nested data_tasks to Arrow representation
                    # category_field = pa.field('category', pa.string())
                    # values_field = pa.field('values', pa.list_(pa.int64()))
                    # struct_type = pa.struct([category_field, values_field])
                    #
                    # # Create a list array containing the structured data_tasks
                    # arrays = [
                    #     pa.array(
                    #         [(item['category'], item['values']) for item in nested_data],
                    #         type=struct_type)
                    # ]
                    # list_array = pa.ListArray.from_arrays(arrays)
                    #
                    # # Displaying the Arrow representation of nested data_tasks
                    # exec_log.info(list_array)
                    # feather.write_feather(list_array, 'data_tasks.feather')
                    #
                    # # for parquet (move to above)
                    # table = pa.Table.from_arrays([list_array], names=['nested_data'])
                    # pq.write_table(table, 'data_tasks.parquet')
                if isinstance(data, pyarrow.Table):
                    arrow_to_feather(data, path=full_path)
                elif isinstance(data, pd.DataFrame):
                    arrow_to_feather(pandas_to_arrow(data), path=full_path)
                else:
                    exec_log.warning(
                        f"Serialization of data_tasks type {type(data)} to feather is not supported."
                    )
            else:
                save_csv(data, path=full_path)
    return data
