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
