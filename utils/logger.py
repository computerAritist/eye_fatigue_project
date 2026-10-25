"""CSV persistence for video measurements, flushed after every completed row."""

import csv
from dataclasses import fields
from pathlib import Path

from utils.measurement_window import MeasurementResult


class MeasurementCSV:
    def __init__(self, path, *, overwrite=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w" if overwrite else "x", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(
            self.file, fieldnames=[field.name for field in fields(MeasurementResult)]
        )
        self.writer.writeheader()
        self.file.flush()

    def write(self, result):
        values = result.as_dict()
        self.writer.writerow({
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in values.items()
        })
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
