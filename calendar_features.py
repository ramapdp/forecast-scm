import datetime

import pandas as pd
import holidays

ID_HOLIDAYS = holidays.country_holidays("ID", years=[2024, 2025])

# Verified against holidays.country_holidays("ID", years=[2024, 2025]) output
# (holidays package version 0.83) and cross-checked with published Indonesian
# government Islamic-calendar dates (see Task 3 of the implementation plan).
# The package's Eid al-Fitr / Eid al-Adha dates for 2024 and 2025 matched the
# original working assumptions exactly, so no corrections were required.
RAMADAN_PERIODS: dict[int, tuple[datetime.date, datetime.date]] = {
    2024: (datetime.date(2024, 3, 11), datetime.date(2024, 4, 9)),
    2025: (datetime.date(2025, 3, 1), datetime.date(2025, 3, 30)),
}
EID_AL_FITR_DATES: dict[int, datetime.date] = {
    2024: datetime.date(2024, 4, 10),
    2025: datetime.date(2025, 3, 31),
}
EID_AL_ADHA_DATES: dict[int, datetime.date] = {
    2024: datetime.date(2024, 6, 17),
    2025: datetime.date(2025, 6, 6),
}
PROXIMITY_WINDOW_DAYS = 14


def day_of_week(date_col: pd.Series) -> pd.Series:
    return date_col.dt.dayofweek


def day_of_month(date_col: pd.Series) -> pd.Series:
    return date_col.dt.day


def month(date_col: pd.Series) -> pd.Series:
    return date_col.dt.month


def is_weekend(date_col: pd.Series) -> pd.Series:
    return date_col.dt.dayofweek >= 5


def is_national_holiday(date_col: pd.Series) -> pd.Series:
    return date_col.dt.date.isin(ID_HOLIDAYS)


def is_ramadan(date_col: pd.Series) -> pd.Series:
    def check(d):
        period = RAMADAN_PERIODS.get(d.year)
        return period is not None and period[0] <= d <= period[1]
    return date_col.dt.date.apply(check)


def days_into_ramadan(date_col: pd.Series) -> pd.Series:
    def compute(d):
        period = RAMADAN_PERIODS.get(d.year)
        if period is None or not (period[0] <= d <= period[1]):
            return float("nan")
        return (d - period[0]).days
    return date_col.dt.date.apply(compute)


def days_until_ramadan(date_col: pd.Series) -> pd.Series:
    def compute(d):
        period = RAMADAN_PERIODS.get(d.year)
        if period is None or d >= period[0]:
            return float("nan")
        return (period[0] - d).days
    return date_col.dt.date.apply(compute)


def is_eid_al_fitr(date_col: pd.Series) -> pd.Series:
    return date_col.dt.date.apply(lambda d: EID_AL_FITR_DATES.get(d.year) == d)


def days_since_eid_al_fitr(date_col: pd.Series) -> pd.Series:
    def compute(d):
        eid_date = EID_AL_FITR_DATES.get(d.year)
        if eid_date is None or d < eid_date:
            return float("nan")
        delta = (d - eid_date).days
        return delta if delta <= PROXIMITY_WINDOW_DAYS else float("nan")
    return date_col.dt.date.apply(compute)


def days_until_eid_al_fitr(date_col: pd.Series) -> pd.Series:
    def compute(d):
        eid_date = EID_AL_FITR_DATES.get(d.year)
        if eid_date is None or d > eid_date:
            return float("nan")
        delta = (eid_date - d).days
        return delta if delta <= PROXIMITY_WINDOW_DAYS else float("nan")
    return date_col.dt.date.apply(compute)
