import logging
import pandas as pd
from pathlib import Path
import pyarrow
import pyarrow.feather as feather
import pyarrow.parquet as pq
from typing import List, Optional, Union

exec_log = logging.getLogger("batten.expose")


def pandas_to_arrow(pandas_table: pd.DataFrame,
                    schema_=None,
                    columns_: Optional[List[str]] = None,
                    ) -> pyarrow.Table:
    arrow_table = pyarrow.Table.from_pandas(
        pandas_table,
        schema=schema_,
        columns=columns_
    )
    exec_log.info(f"Pandas DataFrame converted to Arrow Table with schema\n"
                  f"{arrow_table.schema}")

    # This assertion is complicated by fact that arrow table could have
    # more columns because it will place df index into columns
    # assert pandas_table.shape == arrow_table.shape
    return arrow_table


def arrow_to_pandas(
        arrow_table: pyarrow.Table,
) -> pd.DataFrame:
    df = arrow_table.to_pandas()
    exec_log.info(f"Converted Arrow Table to pandas DataFrame.")
    return df


def dictionary_to_arrow_table(
        data: dict,
) -> pyarrow.Table:
    """Note that pyarrow.array can infer nested structure, while
    pyarrow.Table cannot have differently sized arrays.
    """
    arrow_table = pyarrow.table(data)
    exec_log.info(f"Converted dictionary to Arrow Table.")
    return arrow_table


def dictionary_to_arrow_array(
        data: dict
) -> pyarrow.array:
    """Note that pyarrow.array can infer nested structure, while
        pyarrow.Table cannot have differently sized arrays is None:
        pass
    """
    arrow_array = pyarrow.array(data)
    exec_log.info(f"Converted dictionary to Arrow Array.")
    return arrow_array


def arrow_array_to_disk(
        data: pyarrow.array,
        path: Path
) -> None:
    """ Writes nested data_tasks to Arrow IPC, a raw arrow format.

    Nested data_tasks cannot be serialized to feather or parquet, as these
    formats require a pyarrow Table, which in turn is restrixted to
    equally sized arrays.
    """
    schema = pyarrow.schema([
        pyarrow.field('nums', data.type)
    ])

    with pyarrow.OSFile('arrowdata.arrow', 'wb') as sink:
        with pyarrow.ipc.new_file(sink, schema) as writer:
            batch = pyarrow.record_batch([data], schema=schema)
            writer.write(batch)

# ---------------------------------------------------------------------------
# Arrow Table, variously serialized to files


def arrow_to_parquet(
        arrow_table: pyarrow.Table,
        path: Path,
        prefix: Optional[str] = '',
        subject: str = '',
) -> None:
    if len((path/prefix).suffix) == 0:
        full_path = (path/prefix).with_suffix(".parquet")
    else:
        full_path = path/prefix
    pq.write_table(arrow_table, full_path)
    exec_log.info(f"Saved Arrow Table of {subject}; shape {arrow_table.shape} "
                  f"{arrow_table.column_names} to {full_path.as_posix()};"
                  f"{arrow_table.nbytes/1e3:,.1f} KB")


def arrow_to_feather(
    arrow_table: pyarrow.Table,
        path: Path,
        prefix: Optional[str] = '',
        subject: str = '',
) -> None:
    if len((path/prefix).suffix) == 0:
        full_path = (path/prefix).with_suffix(".parquet")
    else:
        full_path = path/prefix
    feather.write_feather(arrow_table, full_path)
    exec_log.info(f"Saved Arrow Table of {subject}; shape {arrow_table.shape} "
                  f"{arrow_table.column_names} to {full_path.as_posix()};"
                  f"{arrow_table.nbytes/1e3:,.1f} KB")

# ---------------------------------------------------------------------------
# Arrow Table, variously serialized to files


def load_feather(
        directory: str,
        file: str = '',
        destination: str = 'arrow'
) -> Union[pd.DataFrame, pyarrow.Table]:
    full_path = directory/file
    if destination == 'pandas':
        data = feather.read_feather(full_path)
        exec_log.info(
            f"Loaded {data.shape[0]:,} records with {data.shape[1]} columns in a"
            f"DataFrame from {full_path.as_posix()}."
        )
    else:
        data = feather.read_table(full_path)
        exec_log.info(
            f"Loaded {data.num_rows:,} records with {data.num_columns} columns in an"
            f"Arrow Table from {full_path.as_posix()}."
        )
    return data


def load_parquet(
        directory: str,
        file: str = '',
        destination: str = 'parquet'
) -> Union[pd.DataFrame, pyarrow.Table]:
    full_path = directory / file
    if destination == 'pandas':
        data = pq.read_table(full_path).to_pandas()
        exec_log.info(
            f"Loaded {data.shape[0]:,} records with {data.shape[1]} columns in a"
            f"DataFrame from {full_path.as_posix()}."
        )
    else:
        data = pq.read_table(full_path)
        exec_log.info(
            f"Loaded {data.num_rows:,} records with {data.num_columns} columns in an"
            f"Arrow Table from {full_path.as_posix()}."
        )
    return data
