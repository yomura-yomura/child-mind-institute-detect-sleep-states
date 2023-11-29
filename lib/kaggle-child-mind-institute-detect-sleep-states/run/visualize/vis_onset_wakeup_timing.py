import polars as pl
from pandas.tseries.holiday import USFederalHolidayCalendar

import child_mind_institute_detect_sleep_states.data.comp_dataset

cal = USFederalHolidayCalendar()
holidays = cal.holidays()

event_df = (
    child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df(
        "train", as_polars=True
    )
    .with_columns(
        pl.col("timestamp").str.to_datetime()
        # .dt
        # # .convert_time_zone("America/Caracas")
        # .convert_time_zone("America/Chicago")
    )
    .with_columns(
        hour=pl.col("timestamp").dt.hour(),
        weekday=pl.col("timestamp").dt.weekday(),
        date=pl.col("timestamp").dt.date(),
    )
    .with_columns(is_holiday=pl.col("date").is_in(pl.Series(holidays).dt.date()))
    .drop_nulls()
    .collect()
    .to_pandas()
)

event_df["is_valid_holiday"] = event_df["is_holiday"] & event_df["weekday"].isin([1, 5])
import plotly.express as px

fig = px.histogram(
    event_df,
    x="hour",
    color="event",
    barmode="overlay",
    # facet_row="is_holiday",
    facet_row="is_valid_holiday",
    # facet_row="weekday",
    histnorm="probability density",
    category_orders={"weekday": event_df["weekday"].unique()},
)
fig.update_xaxes(dtick=1, range=(event_df["hour"].min(), event_df["hour"].max()))
fig.update_yaxes(range=(0, 0.5))
fig.show()

print(event_df.groupby(["weekday", "event"])["hour"].mean())
