"""
Backlog:
- Basic histogram (without needing to ignore 5k data_tasks point limit); use built-in Counter
and pandas cut or qcut
"""
from pathlib import Path
from typing import Tuple, Union, List, Optional
import logging
import numpy as np
import pandas as pd
import altair as alt
# from scipy.stats import norm
import battenkill.utils.io_ as io

# from alibi_detect.cd import BinCD
# from alibi_detect.utils.data_tasks import create_outlier_batch
# from alibi_detect.models.resnet import DetectorResNet

config = io.load_config_file(Path.cwd() / "config-batten.yaml")
exec_log = logging.getLogger("battenkill.detail")


def configure_chart(
    chart_: alt.Chart,
    chart_title_font_size: int = 14,
    axis_title_font_size: int = 14,
) -> alt.Chart:
    pan_zoom = alt.selection_interval(bind="scales")
    return (
        chart_.configure_axis(
            labelFontSize=axis_title_font_size - 2,
            labelColor="gray",
            tickSize=3,
            titleFontSize=axis_title_font_size,
            titleFontWeight="normal",
            # gridOpacity=0.8,
            # gridDash=[5,5],
            domain=False,
        )
        .configure_view(strokeWidth=0, cursor="crosshair")
        .configure_title(
            anchor="start", fontSize=chart_title_font_size, fontWeight="normal"
        )
    ).add_params(pan_zoom)


def altair_variable_encoder(source: pd.DataFrame, z_: str) -> Optional[str]:
    """
    Provides Altair with the proper encoding for an axis variable in shorthand.
    Consider refactoring using the long-form alt.X('name', type='quantitative'), etc.

    :param source: tabular data_tasks set containing a column to be plotted
    :param z_: column to be plotted
    """
    if source[z_].dtype in [int, float]:
        variable_encoding = z_ + ":Q"  # a continuous real-valued quantity
        exec_log.info(f"Variable {z_} encoded as {variable_encoding}.")
        return variable_encoding
    elif pd.api.types.is_datetime64_ns_dtype(source[z_]):
        variable_encoding = z_ + ":T"  # a time or date value
        return variable_encoding
    elif source[z_].dtype == "object":
        variable_encoding = z_ + ":O"
        return variable_encoding
    else:
        print(
            f"Column {z_} is not correctly formatted; please convert to either int,"
            f" float, datetime, or string (or object)."
        )
    return


def altair_axis_encoding(
    source: pd.DataFrame,
    variables: Union[str, List[str]],
) -> Union[str, List[str]]:
    """
    Manages variable encoding, in case multiple names are listed,
    which arises when we want Altair to display multiple fields upon
    mouseover (tooltip).

    Parameters
    ----------
    source
    variables

    Returns
    -------

    """
    if isinstance(variables, list):
        variables_encoded = []
        for variable in variables:
            variables_encoded.append(altair_variable_encoder(source=source, z_=variable))
        return variables_encoded
    elif isinstance(variables, str):
        return altair_variable_encoder(source, variables)
    else:
        exec_log.warning(
            f"{variables} is neither a string or a list of strings, "
            f"cannot tag with an encoding"
        )
        return variables


# class scatterPlots:
#     def __init__(self):


def altair_ts_scatter(
    source: pd.DataFrame,
    x_: str,
    y_: str,
    tooltip_fld: Optional[Union[str, List[str]]] = None,
    categorical_colors: bool = False,
    _cat: str = config["colors"]["symbol"],
    x_title: str = "",
    y_title: str = "",
    _zero: bool = False,
    _title: str = "",
    h_w: Tuple[str] = (300, 600),
    save: bool = False,
    chart_file_stem: Optional[str] = None,
):
    """
    Using filled circles, plots time series of transaction metrics (amount, volume, etc.)
    on a daily resolution. To the right, plot a frequency distribution of the same variable.

    :param source: tabular data_tasks set containing columns to be plotted
    :param x_: column name containing datetimes or integer denoting day of week or year, week or month, etc.
    :param y_: transaction amount, volume, or other metric
    :param _cat: categorical field used to color-code symbols, or a default single color
    :param x_title: horizontal axis title
    :param y_title: vertical axis title
    :param _zero: whether to scale the vertical axis from zero (True) or on the basis of the range of values (False)
    :param _title: chart title
    :returns: altair graph (json) object
    """
    if not isinstance(tooltip_fld, list):
        tooltip_fld = [tooltip_fld]
    tooltip_encoding = altair_axis_encoding(source, tooltip_fld)

    if not categorical_colors:
        chart = (
            alt.Chart(source)
            .mark_circle(opacity=0.6, color=_cat)
            .encode(
                x=alt.X(
                    altair_axis_encoding(source, x_),
                    title=x_title,
                    axis=alt.Axis(
                        grid=False,
                        ticks=True,
                    ),
                ),
                y=alt.Y(
                    y_ + ":Q",
                    title=y_title,
                    scale=alt.Scale(zero=_zero),
                    axis=alt.Axis(
                        grid=True,
                        ticks=True,
                        domain=False,  # axis line
                        tickMinStep=0.5,
                    ),
                ),
                tooltip=tooltip_encoding,
            )
            .properties(
                title=_title,
                height=h_w[0],
                width=h_w[1],
            )
        )
        # .configure_title(anchor='start') only works with one chart (no layering)
    else:
        chart = (
            alt.Chart(source)
            .mark_circle(opacity=0.6)
            .encode(
                x=alt.X(
                    x_axis_encoding,
                    title=x_title,
                    axis=alt.Axis(
                        grid=False,
                        ticks=True,
                    ),
                ),
                y=alt.Y(
                    y_ + ":Q",
                    title=y_title,
                    scale=alt.Scale(zero=_zero),
                    axis=alt.Axis(grid=True, ticks=True, domain=False),
                ),
                color=_cat + ":N",
                tooltip=tooltip_fld,  # altair_axis_encoding(source, tooltip_fld)
            )
            .properties(
                title=_title,
                height=h_w[0],
                width=h_w[1],
            )
        )
    if save:
        io.save_altair(
            charts=configure_chart(chart),
            cpath=Path.cwd()/config["paths"]["reports"]["charts"],
            chart_file_stem=chart_file_stem,
        )
    return chart


def altair_ts_line(
    source: pd.DataFrame,
    x_: str,
    y_: str,
    tooltip_fld: Optional[str] = None,
    _color: str = config["colors"]["line_1"],
    x_title: str = "",
    y_title: str = "",
    _zero: bool = False,
    _title: str = "",
    h_w: Tuple[str] = (300, 600),
    save_chart: bool = True):
    """
    Using symbols, plots time series of transaction metrics (amount, volume, etc.) on a daily resolution.
    To the right, plot a frequency distribution of the same variable.

    :param source: tabular data_tasks set containing columns to be plotted
    :param x_: column name containing datetimes or integer denoting day of week or year, week or month, etc.
    :param y_: transaction amount, volume, or other metric
    :param x_title: horizontal axis title
    :param y_title: vertical axis title
    :param _zero: whether to scale the vertical axis from zero (True) or on the basis of the range of values (False)
    :param _title: chart title
    :param cat: categorical field used to color-code symbols.
    :returns: altair graph (json) object
    """
    x_axis_encoding = altair_variable_encoder(source, x_)
    ts_line = (
        alt.Chart(source)
        .mark_line(color=_color, strokeWidth=1)
        .encode(
            x=alt.X(
                x_axis_encoding,
                title=x_title,
                axis=alt.Axis(
                    grid=False,
                    ticks=True,
                ),
            ),
            y=alt.Y(
                y_ + ":Q",
                title=y_title,
                scale=alt.Scale(zero=_zero),
                axis=alt.Axis(
                    grid=True,
                    ticks=True,
                    domain=False,  # axis line
                    tickMinStep=0.5,
                    orient="right",
                ),
            ),
            tooltip=tooltip_fld,
        )
        .properties(
            title=_title,
            height=h_w[0],
            width=h_w[1],
        )
    )
    if save_chart:
        io.save_altair(
            charts=configure_chart(ts_line),
            cpath=fields["path"],
            chart_file_stem=f"{fields['what']}_{fields['subject']}",
        )
    else:
        return ts_line


def altair_area_between(
    source: pd.DataFrame,
    x_: str,
    y_: str,
    y2_: str,
    x_title: str = "",
    y_title: str = "",
    _color: str = config["colors"]["grey_Light"],
    _zero: bool = False,
    _title: str = "",
    h_w: Tuple[str] = (300, 600),
):
    """
    Shades area between two curves.

    :param source: tabular data_tasks set containing columns to be plotted
    :param x_: horizontal variable
    :param y_: lower vertical variable
    :param y2_: upper vertical variable
    :param x_title: horizontal axis title
    :param y_title: vertical axis title
    :param _zero: whether to scale the vertical axis from zero (True) or on the basis of the range of values (False)
    :param _title: chart title
    """
    return (
        alt.Chart(source)
        .mark_area(opacity=0.8, color=_color)
        .encode(
            x=alt.X(
                x_,
                title=x_title,
                axis=alt.Axis(
                    grid=False,
                    ticks=True,
                ),
            ),
            y=alt.Y(
                y_,
                title=y_title,
                scale=alt.Scale(zero=_zero),
                axis=alt.Axis(grid=True, ticks=True, domain=False),  # axis line
            ),
            y2=y2_,
        )
        .properties(
            title=_title,
            height=h_w[0],
            width=h_w[1],
        )
    )


def altair_area_under(
    source: pd.DataFrame,
    x_: str,
    y_: str,
    x_title: str = "",
    y_title: str = "",
    _color: str = "red",
    _zero: bool = False,
    _title: str = "",
    h_w: Tuple[str] = (300, 600),
):
    """
    Shades area under one curve.

    :param source: tabular data_tasks set containing columns to be plotted
    :param x_: horizontal variable
    :param y_: vertical variable
    :param source: tabular data_tasks set containing columns to be plotted
    :param x_title: horizontal axis title
    :param y_title: vertical axis title
    :param _zero: whether to scale the vertical axis from zero (True) or on the basis of the range of values (False)
    :param _title: chart title
    """
    return (
        alt.Chart(source)
        .mark_area(
            opacity=0.5,
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(
                        color=config["colors"]["shade_translucent"], offset=0
                    ),
                    alt.GradientStop(
                        color=config["colors"]["shade_background"], offset=1
                    ),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
        )
        .encode(
            x=alt.X(
                x_,
                title=x_title,
                axis=alt.Axis(
                    grid=False,
                    ticks=True,
                ),
            ),
            y=alt.Y(
                y_,
                title=y_title,
                scale=alt.Scale(zero=_zero),
                axis=alt.Axis(grid=True, ticks=True, domain=False),  # axis line
            ),
        )
        .properties(
            title=_title,
            height=h_w[0],
            width=h_w[1],
        )
    )


def freq_distn(
    data: Union[list, np.ndarray, pd.Series],
    bin_title: str,
    count_title: str,
    log_transform: bool = False,
    mom1: float = None,
    mom2: float = None,
    multiple: float = 2,
    _bins: int = 10,
    _mark: str = "bar",
    _range: Optional[Tuple[float, float]] = None,
    _density: bool = True,
    col: str = config["colors"]["line_1"],
    _title: str = "",
    x_title: str = "",
    y_title: str = "count",
    fit_distn: bool = False,
    shade_region: bool = False,
    h_w: Tuple[int] = (300, 300),
    save_chart: bool = True,
    **fields,
) -> alt.Chart:
    """
    Groups observations into bins and produces a scatter plot of frequency versus
    the midpoints of those bins. Optionally fits the distribution.

    :param data: 1D data_tasks
    :param z_: column header
    :param mom1: 1st moment (mean) of fitted distribution
    :param mom2: 2nd moment (standard deviation) of fitted distribution
    :param multiple: $\mu + m\cdot\sigma$
    :param _bins: number of bins in which to put equal number of observations
    :param _density: whether to normalize the frequency distribution
    :param col: color
    :param x_title: horizontal axis label
    :param fit_distn: whether to plot a fitted distribution with the empirical distribution
    :param shade_region: whether to shade the region above the threshold defined by `mom1`, `mom2`, and `multiple`
    """

    def bin_edges_to_strings(bin_edges):
        intervals = []
        for i in range(len(bin_edges)):
            interval_str = f"{bin_edges[i]} to {bin_edges[i+1]}"
            intervals.append(interval_str)
        return intervals

    if log_transform:
        output = np.zeros(len(data))
        data = np.log10(data, out=output, where=data > 0)

    hist, bin_edges_ = np.histogram(data, _bins, _range, density=_density)
    # midpoints = [
    #     (bin_edges[i + 1] + bin_edges[i]) / 2 for i in range(bin_edges.shape[0] - 1)
    # ]

    cdata = pd.DataFrame(
        {bin_title: bin_edges_to_strings(bin_edges_), count_title: hist}
    )

    if _mark == "bar":
        charter = BarChart(data=cdata)
        c1 = charter.vbar(
            x_=bin_title + ":O",
            y_=count_title + ":Q",
            x_title=x_title,
            y_title=y_title,
            h_w=h_w,
        )
    elif _mark == "point":
        c1 = (
            alt.Chart(cdata)
            .mark_circle(opacity=0.8, color=col)
            .encode(
                x=alt.X(bin_title, title=x_title, scale=alt.Scale(zero=False)),
                y=alt.Y(count_title, title=y_title),
            )
            .properties(
                title=_title,
                height=h_w[0],
                width=h_w[1],
            )
        )

    if fit_distn:
        fit_df = pd.DataFrame(
            {"fit_x": np.linspace(1.1 * bin_edges_.min(), 1.1 * bin_edges_.max())}
        )
        fit_df = fit_df.assign(
            fit_y=norm.pdf(fit_df.fit_x, mom1, mom2)
        )  # moments produced elsewhere
        c2 = altair_ts_line(fit_df, "fit_x", "fit_y")
        if shade_region:
            fit_portion = fit_df[fit_df.fit_x > (mom1 + multiple * mom2)]
            c3 = altair_area_under(fit_portion, x_="fit_x", y_="fit_y")
            c_final = (c1 + c2 + c3).configure_title(anchor="start")
        else:
            c_final = (c1 + c2).configure_title(anchor="start")
    else:
        c_final = c1

    if save_chart:
        io.save_altair(
            charts=c_final.configure_title(anchor="start")
            .configure_axis(
                labelFontSize=16, titleFontSize=16, titleFontWeight="normal"
            )
            .configure_view(strokeWidth=0),
            cpath=Path(config["paths"]["reports"]["charts"]) / fields["session_id"],
            chart_file_stem=(
                f"{fields['what']}_{fields['subject']}"
            ),  # _{fields['channel']}
        )


def altair_freq_distn90(
    source: pd.DataFrame,
    y_: str,
    log_x: bool = False,
    log_y: bool = False,
    h_w: Tuple[str] = (300, 100),
):
    """
    Plot a frequency distribution of the same variable, rotated 90 degrees

    :param y_: transaction amount, volume, or other metric
    :param log_x: exposes extreme values by linearizing (log-transforming) skewed data_tasks
    :param log_y: exposes extreme values by log-transforming skewed data_tasks

    :returns: altair graph (json) object
    """
    if log_y:
        source = log_transform(source, y_)
    return (
        alt.Chart(source)
        .mark_bar()
        .encode(
            x="count()",
            y=alt.Y(y_ + ":Q", bin=alt.Bin(maxbins=30)),
            # color='species:N'
        )
        .properties(height=h_w[0], width=h_w[1])
    )


def compound_ts_distribution(chart1, chart2, orientation: str = "horizontal"):
    """
    Juxtaposes two altair charts
    """
    if orientation == "horizontal":
        return alt.hconcat(chart1, chart2).configure_view(strokeWidth=0)
    elif orientation == "vertical":
        return alt.vconcat(chart1, chart2).configure_view(strokeWidth=0)
    else:
        return f"Orientation was misspecified as {orientation}; options include horizontal and vertical."


def compound_prediction_residual(
    data: pd.DataFrame, plot_residuals: bool = False, save_chart: bool = True, **fields
) -> None:
    """Plots observations, predictions, upper confidence interval in one chart,
    optionally with color-encoded observations. Below that graph, plots model residuals
    and their distribution.  Primary purpose is to visualize and evaluate the performance of one model.

    Parameters
    ----------
    data
        Fields containing data_tasks to plot.
    plot_residuals
        Whether the data_tasks contain prediction residuals to plot.
    save_chart
        Whether to save the chart.
    fields
        A kwargs-like dict containing the names of columns and descriptive metadata.

    Returns
    -------
    None
    """
    # For panning and zooming, create a selection object to capture user mouse input
    pan_zoom = alt.selection_interval(bind="scales")

    observations = altair_ts_scatter(
        data,
        x_=fields["t"],
        y_=fields["y"],
        tooltip_fld=fields["t"],
        categorical_colors=fields["categorical_colors"],
        _cat=fields["color_encoding"],
        _title=f"{fields['subject']}, {fields['method']}",  # , {fields['channel']}
    )
    model_predictions = altair_ts_line(
        data,
        x_=fields["t"],
        y_=fields["y_hat"],
    )
    confidence_interval_upper = altair_area_between(
        data,
        x_=fields["t"],
        y_=fields["y_hat"],
        y2_=fields["ci"],
    )
    if plot_residuals:
        # Compute residuals and their distribution; optionally, save within a multi-chart
        residuals = altair_ts_line(
            data, x_=fields["t"], y_=fields["residuals"], _color="gray", h_w=(100, 600)
        ) + alt.Chart(  # line showing average error
            data
        ).mark_rule(
            color=config["colors"]["line_secondary"], strokeDash=[5, 5]
        ).encode(
            y="mean(" + fields["residuals"] + "):Q"
        )
        residual_distribution = altair_freq_distn90(data, "residuals", h_w=(100, 100))

        if save_chart:
            io.save_altair(
                (
                    (confidence_interval_upper + model_predictions + observations)
                    & (residuals | residual_distribution)
                )
                .configure_view(strokeWidth=0)
                .configure_axis(
                    labelFontSize=16, titleFontSize=16, titleFontWeight="normal"
                )
                .add_selection(pan_zoom),
                Path(config["paths"]["reports"]["charts"]) / fields["session_id"],
                chart_file_stem=(
                    f"{fields['method']}_{fields['subject']}.html"
                ),  # _{fields['channel']}
            )
            return  # prevents overwrite after redundant check of save_chart

    # If the residuals are not plotted, save a simpler multi-chart
    if save_chart:
        io.save_altair(
            charts=(confidence_interval_upper + observations + model_predictions)
            .configure_view(strokeWidth=0)
            .configure_axis(
                labelFontSize=16, titleFontSize=16, titleFontWeight="normal"
            )
            .add_selection(pan_zoom),
            cpath=Path(config["paths"]["reports"]["charts"]) / fields["session_id"],
            chart_file_stem=(
                f"{fields['method']}_ts_{fields['subject']}"
            ),  # _{fields['channel']}
        )


def compound_anomaly(data: pd.DataFrame, save_chart: bool = True, **kwargs) -> None:
    """Plots color-encoded observations.  Primary purpose is to visualize
    the final classification based on one or more models and/or rules.

    Why not just use the basic ts function?  What else might we want to include in this
    'dashboard'?

    Parameters
    ----------
    data
        Fields containing data_tasks to plot.
    fields
        A kwargs-like dict containing the names of columns and descriptive metadata.

    Returns
    -------
    None
    """
    # For panning and zooming, create a selection object to capture user mouse input
    pan_zoom = alt.selection_interval(bind="scales")

    observations = altair_ts_scatter(
        data,
        x_=kwargs["t"],
        y_=kwargs["y"],
        tooltip_fld=kwargs["t"],
        _cat="anomaly_flag:N",
        _title=f"{kwargs['subject']}",  # , {kwargs['channel']}
    )
    if save_chart:
        io.save_altair(
            (observations + model_predictions + confidence_interval_upper)
            .configure_view(strokeWidth=0)
            .add_selection(pan_zoom),
            Path(kwargs["session_id"]) / config["paths"]["reports"]["charts"],
            chart_file_stem=(f"dash_{kwargs['subject']}.html"),  # _{kwargs['channel']}
        )


def weight_plot(
    data: pd.DataFrame,
    x_field: str = "beta",
    x_title: str = "Coefficient",
    y_field: str = "Predictor",
    ci: Tuple = ("beta_lci", "beta_uci"),
    save_chart: bool = True,
    **fields,
) -> None:
    points = (
        alt.Chart(data)
        .mark_point(filled=True, color="black")
        .encode(
            x=alt.X(x_field, title=x_title),
            y=alt.Y(
                y_field, sort=alt.EncodingSortField(field=x_field, order="descending")
            ),
        )
        .properties(
            title=f"Regression weights: {fields['subject']}", width=400, height=250
        )
    )
    error_bars = points.mark_rule().encode(
        x=ci[0],
        x2=ci[1],
    )
    if save_chart:
        io.io.save_altair(
            charts=(points + error_bars)
            .configure_view(strokeWidth=0)
            .configure_axis(
                labelFontSize=16, titleFontSize=16, titleFontWeight="normal"
            ),
            cpath=Path(config["paths"]["reports"]["charts"]) / fields["session_id"],
            chart_file_stem=(
                f"{fields['method']}_weightPlot_{fields['subject']}"
            ),  # _{fields['channel']}
        )


def compound_fed_repo_rates(
    data: List[pd.DataFrame],
    # fields: List[List[str]],
    save_chart=False,
):
    """Plots the range of the Federal Reserve's funds rate target, the effective rate of
    same, and the Secured Overnight Financing Rate.

    Parameters
    ----------
    data
        The data_tasks are programmatically downloaded as single series (this may be changed
        to a SeriesCollection), and so provided as a list in the order implied.
    save_chart

    Returns
    -------

    """
    effr = altair_ts_scatter(
        data[0], x_="date", y_="value", y_title="Rate (percent)", tooltip_fld="value"
    )
    sofr = altair_ts_line(
        data[1], x_="date", y_="value", _color=config["colors"]["line_secondary"]
    )
    ffrt_range = altair_area_between(data[2], x_="date", y_="DFEDTARL", y2_="DFEDTARU")

    if save_chart:
        io.io.save_altair(
            charts=(ffrt_range + effr + sofr)
            .configure_view(strokeWidth=0)
            .configure_axis(
                labelFontSize=16, titleFontSize=16, titleFontWeight="normal"
            ),
            cpath=Path(config["paths"]["reports"]["charts"]),
            chart_file_stem="fed_repo_rate_chart_2023",
        )
    return ffrt_range + effr + sofr


def plot_bars(
    data: pd.DataFrame,
    z: str,
    save_chart: bool = False,
    chart_file_stem: str = "",
    h_w: Tuple[str] = (300, 600),
):
    """

    Parameters:
    - data_tasks: Pandas DataFrame with column

    Returns:
    - Altair Chart
    """
    # Create Altair Chart
    chart = alt.Chart(data)

    # Plot candlestick bars
    bars = (
        chart.mark_bar()
        .encode(
            x="date:T",
            y=z + ":Q",
            color=alt.condition("datum.z < 0", alt.value("#ae1325"), alt.value("grey")),
        )
        .properties(
            height=h_w[0],
            width=h_w[1],
        )
    )
    if save_chart:
        io.save_altair(
            charts=bars.configure_view(strokeWidth=0).configure_axis(
                labelFontSize=16, titleFontSize=16, titleFontWeight="normal"
            ),
            cpath=Path(config["paths"]["reports"]["charts"]),
            chart_file_stem=chart_file_stem,
        )
    return bars


def plot_candlesticks(data):
    """
    Plot stock price candlesticks using Altair.

    Parameters:
    - data_tasks: Pandas DataFrame with columns ['date', 'open', 'high', 'low', 'close']

    Returns:
    - Altair Chart
    """
    # Create Altair Chart
    chart = alt.Chart(data)

    # Plot candlestick bars
    candlesticks = chart.mark_bar().encode(
        x="date:T",
        y="low:Q",
        y2="high:Q",
        color=alt.condition(
            "datum.open <= datum.close", alt.value("#06982d"), alt.value("#ae1325")
        ),
    )

    # Plot ticks for open and close prices
    ticks = chart.mark_tick().encode(
        x="date:T",
        y="open:Q",
        size=alt.value(5),
        color=alt.value("black"),
    ) + chart.mark_tick().encode(
        x="date:T",
        y="close:Q",
        size=alt.value(5),
        color=alt.value("black"),
    )

    # Combine candlesticks and ticks
    chart = candlesticks + ticks

    return chart


# Example usage:
# Assume 'stock_data' is your DataFrame with columns ['date', 'open', 'high', 'low', 'close']
# chart = plot_candlesticks(stock_data)
# chart.show()


# def plot_time_series_with_changepoints(time_series_data):
#     # Assuming 'time_series_data' is a pandas DataFrame with a column 'timestamp' and 'value'
#
#     # Create an Alibi-detect changepoint detector
#     model = DetectorResNet(
#         threshold=None)  # You can adjust the threshold based on your data_tasks
#     cd = BinCD(model)
#
#     # Convert time_series_data to a numpy array
#     data_array = time_series_data['value'].values.reshape(-1, 1)
#
#     # Create an outlier batch for Alibi-detect
#     batch = create_outlier_batch(data_array)
#
#     # Detect changepoints
#     explanation = cd.explain(batch)
#
#     # Extract changepoints
#     changepoints = explanation['data_tasks']['is_change'].reshape(-1)
#
#     # Add changepoints to the original DataFrame
#     time_series_data['changepoint'] = changepoints.astype(int)
#
#     # Plot the time series with changepoints using Altair
#     chart = alt.Chart(time_series_data).mark_line().encode(
#         x='timestamp:T',
#         y='value:Q',
#         color='changepoint:N'
#     ).properties(width=600, height=300)
#
#     # Customize the chart to highlight changepoints with dots
#     chart = chart.mark_circle(size=60, opacity=1, color='red').encode(
#         x='timestamp:T',
#         y='value:Q',
#         tooltip=['timestamp:T', 'value:Q'],
#     )
#
#     return chart
#
#
# # Example usage:
# # Create a sample time series DataFrame with 'timestamp' and 'value' columns
# sample_data = pd.DataFrame({
#     'timestamp': pd.date_range(start='2022-01-01', end='2022-01-10'),
#     'value': [10, 12, 15, 8, 7, 20, 18, 22, 10, 12]
# })
#
# # Plot the time series with changepoints
# plot = plot_time_series_with_changepoints(sample_data)
#
# # Show the plot
# plot.show()


class BarChart:
    def __init__(
        self,
        source: pd.DataFrame,
        x_: str,
        y_: str,
        orientation: str,
        *,
        autoencode: bool = False,
        pareto: Optional[Tuple[str]] = None,
        label_field: Optional[str] = None,
        label_alignment: str = "left",
        label_offset: int = 5,
        plot_area_pad: float = 1.0,
        min_height: float = 1.0,
        _color: str = config["colors"]["line_1"],
        color_light: str = "white",
        color_dark: str = "black",
        x_title: Optional[str] = None,
        y_title: Optional[str] = None,
        _title: Optional[str] = None,
        x_grid: bool = False,
        y_grid: bool = False,
        tick_toggle: bool = False,
        y_tick_step: float = 1.0,
        tick_labels: bool = True,
        h_w: Tuple[int] = (300, 600),
        save_chart: bool = False,
    ):
        """

        Parameters
        ----------
        source
            Pandas DataFrame containing the labels and heights of the bars to be plotted.
        x_
            Variable plotted on the horizontal axis; can be categorical, if a vertical bar chart.
            If 'autoencode' is 'False', the string must be appended with the Altair dtype specifier;
            e.g., "growth_rate:* and customer_segment: N
        y_
            Variable plotted on the vertical axis; can be categorical, if a horizontal bar
        autoencode
            Whether to call a function that appends an Altair dtype specifier on the basis of its attempt to determine the data_tasks type of the variable.
        orientation
            Whether bar height extends vertically or horizontally, which determines how other plot settings are implemented.
        pareto
            If not 'None", '(-x) will sort horizontal bars in descending order of the bar
            height (top to bottom); '(-y)' will similarly sort vertical bars (left to right). label_field
        label_field
            Usually x_ or y_', the variable whose values will be used to label the end of the bars.
        label_alignment
            'left' or 'right', determines the directionality of ball labels.
        label_offset
            Determines where bar labels begin relative to the ends of bars. plot_area_pad
        plot_area_pad
            To accommodate bar labels; use a number greater than 1; e.g., 1.2 to extend plot area by 20% more than the longest bar.
        min_height
           Threshold that determines _color
        _color
            Color of the bars.
        color_light
            Color of the bar labels.
        color_dark
            Alternative bar label color; e.g., if you want to change the color depending on the "label_alignment".
        x_title
            Title of the horizontal axis.
        y_title
            Title of the vertical axis.
        x grid
            Whether to display vertical grid lines.
        y_grid
            Whether to display horizontal grid lines.
        tick_toggle
            Whether to display tick marks on both or neither axis. y_tick_step
        y_tick_step
            Minimum
        tick_Labels
            Whether to label the tick （positions）
        h_w
            Height and width of the chart.
        save_chart
            Whether to save the chart to a png file.
        fields
            Chart file name parts and path to save the png to.
        """
        self.source = source
        self.x_ = x_
        self.y_
        self.orientation
        self.autoencode = False
        self.pareto = pareto
        self.label_field = label_field
        self.label_alignment = "left"
        self.label_offset = 5
        self.plot_area_pad = 1.0
        self.min_height = 1.0
        self._color = config["colors"]["line_1"]
        self.color_light = "white"
        self.color_dark = "black"
        self.x_title = x_title
        self.y_title = y_title
        self._title = _title
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.tick_toggle = tick_toggle
        self.y_tick_step: float = 1.0
        self.tick_labels = tick_labels
        self.h_w: Tuple[int] = (300, 600)
        self.save_chart = save_chart

        if not self.label_field:
            self.source.loc[:, "dummy"] = ""
            self.label_field = "dummy"

        if not self.autoencode:
            self.x_ = altair_axis_encoding(self.source, self.x_)
            self.y_ = altair_axis_encoding(self.source, self.y_)

    def combine_plot_with_labels(self, base):
        return base.mark_bar(color=self._color) + base.mark_text(
            align=self.label_alignment, dx=self.label_offset, color=self.color_dark
        )

    def hbar(self, **fields) -> alt.Chart:
        """Two reasons to have separate horizontal and vertical bar functions:
        1. Horizontal (vertical) bars have a scaled x (y) axis
        2. If the bars are to be sorted, the horizontal (vertical) axis is sorted.
        """
        base = (
            alt.Chart(self.source)
            .encode(
                x=alt.X(
                    self.x_,
                    title=self.x_title,
                    scale=alt.Scale(
                        domain=[0, self.plot_area_pad * self.source[self.x_[:-2]].max()]
                    ),
                    axis=alt.Axis(
                        grid=self.x_grid,
                        ticks=self.tick_toggle,
                        domain=False,
                        labels=self.tick_labels,
                        labelAngle=0,
                    ),
                ),
                y=alt.Y(
                    self.y_,
                    title=self.y_title,
                    axis=alt.Axis(
                        grid=self.y_grid,
                        ticks=self.tick_toggle,
                        domain=False,
                        tickMinStep=self.y_tick_step,
                    ),
                    sort=self.pareto,
                ),
                text=self.label_field,
            )
            .properties(
                title=self._title,
                height=self.h_w[0],
                width=self.h_w[1],
            )
        )

        chart = self.combine_plot_with_labels(base)

        if self.save_chart:
            io.save_altair(
                charts=configure_chart(chart),
                cpath=fields["path"],
                chart_file_stem=f"{fields['what']}_{fields['subject']}",
            )
        return chart

    def vbar(self, **fields) -> alt.Chart:
        """Two reasons to have separate horizontal and vertical bar functions:
        1. Horizontal (vertical) bars have a scaled x (y) axis
        2. If the bars are to be sorted, the horizontal (vertical) axis is sorted.
        """
        base = (
            alt.Chart(self.source)
            .encode(
                x=alt.X(
                    self.x_,
                    title=self.x_title,
                    axis=alt.Axis(
                        grid=self.x_grid,
                        ticks=self.tick_toggle,
                        domain=False,
                        labels=self.tick_labels,
                        labelAngle=0,
                    ),
                    sort=self.pareto,
                ),
                y=alt.Y(
                    self.y_,
                    title=self.y_title,
                    scale=alt.Scale(
                        domain=[0, self.plot_area_pad * self.source[self.y_[:-2]].max()]
                    ),
                    axis=alt.Axis(
                        grid=self.y_grid,
                        ticks=self.tick_toggle,
                        domain=False,
                        tickMinStep=self.y_tick_step,
                    ),
                ),
                text=self.label_field,
            )
            .properties(
                title=self._title,
                height=self.h_w[0],
                width=self.h_w[1],
            )
        )

        chart = self.combine_plot_with_labels(base)

        if self.save_chart:
            io.save_altair(
                charts=configure_chart(chart),
                cpath=fields["path"],
                chart_file_stem=f"{fields['what']}_{fields['subject']}",
            )
        return chart
