import csv
import datetime

DATE_FORMAT = "%d %b %Y"


def parse_tanggal(value: str) -> datetime.date:
    return datetime.datetime.strptime(value.strip(), DATE_FORMAT).date()
