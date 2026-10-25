"""Presentation-only reports built from the unchanged 17-column measurements."""

import argparse
from contextlib import ExitStack
import csv
from dataclasses import dataclass, fields
import math
from pathlib import Path
import sys
import unicodedata

from utils.logger import MeasurementCSV
from utils.measurement_window import MeasurementResult


SUMMARY_COLUMNS = (
    "측정 구간", "측정 시간(초)", "눈 깜빡임 횟수(회)",
    "분당 깜빡임 환산 횟수(회/분)", "평균 눈 감김 시간(ms)",
    "눈 뜸 크기 감소율(%)", "얼굴 검출 성공률(%)", "측정 상태",
)
METRICS = (
    ("blink_rate", "Blink rate over video time", "Blinks per minute"),
    ("blink_duration", "Mean eye closure duration over video time", "Mean closure duration (ms)"),
    ("eye_reduction", "Normalized eye contour area reduction over video time", "Eye area reduction (%)"),
)


@dataclass(frozen=True)
class ReportPaths:
    raw: Path
    summary: Path
    graphs: tuple[Path, ...]

    @property
    def all_paths(self):
        return (self.raw, self.summary, *self.graphs)


def report_paths(csv_path):
    path = Path(csv_path)
    stem = path.stem if path.suffix.lower() == ".csv" else path.name
    for suffix in ("_raw", "_summary"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    if not stem:
        raise ValueError("Output filename must include a name before _raw/_summary.")
    return ReportPaths(
        path.with_name(f"{stem}_raw.csv"), path.with_name(f"{stem}_summary.csv"),
        tuple(path.with_name(f"{stem}_{name}.png") for name, _, _ in METRICS),
    )


def check_outputs(paths, *, overwrite=False, source=None):
    """Check all artifacts before opening any writer or replacing a source."""
    for path in paths:
        if source is not None and (
            path.resolve() == Path(source).resolve()
            or (path.exists() and Path(source).exists() and path.samefile(source))
        ):
            raise ValueError(f"Output must not replace the input file: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}; use --overwrite to replace reports.")


def interval_label(result):
    def clock(seconds):
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return (f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours
                else f"{minutes:02d}:{seconds:02d}")
    return f"{clock(result.start_sec)}~{clock(result.end_sec)}"


def blink_rate(result):
    # Use unrounded duration, including the final partial window.
    return result.blink_count * 60 / result.duration_sec if result.duration_sec > 0 else None


def measurement_status(result):
    if not result.measurement_valid:
        return "유효하지 않음 (부분 구간)" if result.is_partial else "유효하지 않음"
    return "부분 구간" if result.is_partial else "정상"


def number(value, decimals):
    return f"{value:.{decimals}f}" if value is not None and math.isfinite(value) else ""


def summary_values(result):
    return (
        interval_label(result), number(result.duration_sec, 2), str(result.blink_count),
        number(blink_rate(result), 1), number(result.mean_blink_duration_ms, 1),
        number(result.mean_eye_opening_reduction_pct, 2),
        number(result.face_detection_success_pct, 1), measurement_status(result),
    )


class SummaryCSV:
    def __init__(self, path, *, overwrite=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w" if overwrite else "x", newline="", encoding="utf-8-sig")
        self.writer = csv.writer(self.file)
        self.writer.writerow(SUMMARY_COLUMNS)
        self.file.flush()

    def write(self, result):
        self.writer.writerow(summary_values(result))
        self.file.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.file.close()


def terminal_table(results):
    headers = ("측정 구간", "깜빡임(회/분)", "평균 감김(ms)", "눈 크기 감소율(%)", "상태")
    rows = [headers]
    for result in results:
        values = summary_values(result)
        rows.append((values[0], values[3] or "N/A", values[4] or "N/A",
                     values[5] or "N/A", values[7]))
    if len(rows) == 1:
        return "측정 결과 없음 (보정 완료 후 측정 구간이 없습니다.)"

    def width(text):
        return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)

    widths = [max(width(row[column]) for row in rows) for column in range(len(headers))]
    lines = [" | ".join(cell + " " * (size - width(cell))
                        for cell, size in zip(row, widths)) for row in rows]
    lines.insert(1, "-+-".join("-" * size for size in widths))
    return "\n".join(lines)


def metric_value(result, metric):
    if metric == "blink_rate":
        return blink_rate(result)
    if metric == "blink_duration":
        return result.mean_blink_duration_ms
    if metric == "eye_reduction":
        return result.mean_eye_opening_reduction_pct
    raise ValueError(f"Unknown graph metric: {metric}")


def metric_figure(results, metric, title, ylabel):
    # FigureCanvasAgg avoids GUI dependencies on WSL/headless machines.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    figure = Figure(figsize=(10, 4.8), dpi=150)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    xs, ys, full, partial, invalid = [], [], [], [], []
    for result in results:
        midpoint = (result.start_sec + result.end_sec) / 120
        value = metric_value(result, metric)
        usable = result.measurement_valid and value is not None and math.isfinite(value)
        xs.append(midpoint)
        ys.append(value if usable else math.nan)
        if result.is_partial:
            axis.axvspan(result.start_sec / 60, result.end_sec / 60,
                         facecolor="#f9b44c", alpha=0.13, hatch="///", edgecolor="#d88610")
        if not usable:
            invalid.append(midpoint)
            axis.axvspan(result.start_sec / 60, result.end_sec / 60,
                         facecolor="#d34d4d", alpha=0.13)
        elif result.is_partial:
            partial.append((midpoint, value))
        else:
            full.append((midpoint, value))
    axis.plot(xs, ys, color="#587a99", linewidth=1.7)
    for points, color, marker in ((full, "#247d94", "o"), (partial, "#d88610", "s")):
        if points:
            axis.scatter(*zip(*points), color=color, marker=marker, s=52, zorder=3)
    if invalid:
        # A marker in axis coordinates is a status annotation, never a fabricated zero.
        axis.scatter(invalid, [0.05] * len(invalid), marker="x", color="#b92c38", s=65,
                     transform=axis.get_xaxis_transform(), zorder=4)
    if results:
        axis.set_xlim(results[0].start_sec / 60, results[-1].end_sec / 60)
    else:
        axis.text(0.5, 0.5, "No measurement windows (calibration incomplete)",
                  ha="center", va="center", transform=axis.transAxes)
    if not any(math.isfinite(value) for value in ys):
        axis.set_ylim(0, 1)
        if results:
            axis.text(0.5, 0.5, "No valid measurements for this metric",
                      ha="center", va="center", transform=axis.transAxes)
    axis.set(title=title, xlabel="Video measurement time (minutes)", ylabel=ylabel)
    axis.grid(alpha=0.22)
    axis.legend(handles=[
        Line2D([], [], color="#247d94", marker="o", linestyle="none", label="Valid full window"),
        Line2D([], [], color="#d88610", marker="s", linestyle="none", label="Valid partial window"),
        Patch(facecolor="#f9b44c", alpha=0.3, hatch="///", label="Partial interval"),
        Line2D([], [], color="#b92c38", marker="x", linestyle="none", label="Invalid (excluded)"),
    ], loc="best", fontsize=8)
    figure.tight_layout()
    return figure


def save_graphs(results, paths):
    for (metric, title, ylabel), path in zip(METRICS, paths):
        figure = metric_figure(results, metric, title, ylabel)
        try:
            figure.savefig(path, format="png")
        finally:
            figure.clear()


class AnalysisReports:
    """Write both CSVs per window; retain only window results for final graphs."""

    def __init__(self, csv_path, *, overwrite=False):
        self.paths = report_paths(csv_path)
        check_outputs(self.paths.all_paths, overwrite=overwrite)
        self.rows = []
        self._stack = ExitStack()
        try:
            self.raw = self._stack.enter_context(MeasurementCSV(self.paths.raw, overwrite=overwrite))
            self.summary = self._stack.enter_context(SummaryCSV(self.paths.summary, overwrite=overwrite))
        except BaseException:
            self._stack.close()
            raise

    def write(self, result):
        self.raw.write(result)  # Preserve the original writer and its six-decimal precision.
        self.rows.append(result)
        self.summary.write(result)

    def finalize(self):
        print(terminal_table(self.rows))
        save_graphs(self.rows, self.paths.graphs)
        print(f"[REPORT] raw={self.paths.raw}; summary={self.paths.summary}")
        for path in self.paths.graphs:
            print(f"[GRAPH] {path}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stack.close()


def read_measurements(path):
    """Read existing raw files without importing detectors or running inference."""
    integer_fields = {"window_index", "blink_count", "duration_sample_count",
                      "eor_sample_count", "frame_count", "face_detected_count"}
    bool_fields = {"calibrated", "is_partial", "measurement_valid"}
    text_fields = {"user_id", "end_reason"}
    nullable_fields = {"mean_blink_duration_ms", "mean_eye_opening_reduction_pct"}
    columns = {field.name for field in fields(MeasurementResult)}
    results = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not columns.issubset(reader.fieldnames or []):
            raise ValueError("Expected the original 17-column measurement CSV.")
        for row_number, row in enumerate(reader, start=2):
            values = {}
            try:
                for key in columns:
                    value = row[key]
                    if key in bool_fields:
                        if value not in {"True", "False"}:
                            raise ValueError(f"Invalid boolean {key}: {value}")
                        values[key] = value == "True"
                    elif key in integer_fields:
                        values[key] = int(value)
                    elif key in text_fields:
                        values[key] = value
                    elif key in nullable_fields and value == "":
                        values[key] = None
                    else:
                        values[key] = float(value)
                        if not math.isfinite(values[key]):
                            raise ValueError(f"Non-finite value in {key}")
                results.append(MeasurementResult(**values))
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid CSV row {row_number}: {error}") from error
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create summary CSV/PNGs from an existing raw CSV.")
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("--output", type=Path, help="Output base CSV name (default: input base)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        rows = read_measurements(args.raw_csv)
        paths = report_paths(args.output or args.raw_csv)
        check_outputs((paths.summary, *paths.graphs), overwrite=args.overwrite, source=args.raw_csv)
        with SummaryCSV(paths.summary, overwrite=args.overwrite) as writer:
            for row in rows:
                writer.write(row)
        print(terminal_table(rows))
        save_graphs(rows, paths.graphs)
        print(f"[REPORT] source={args.raw_csv}; summary={paths.summary}")
        for path in paths.graphs:
            print(f"[GRAPH] {path}")
        return 0
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
