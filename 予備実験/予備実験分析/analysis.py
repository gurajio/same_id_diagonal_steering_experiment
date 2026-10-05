from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = BASE_DIR / "output" / ".matplotlib-cache"
XDG_CACHE_DIR = BASE_DIR / "output" / ".cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit
except ImportError:  # The script still writes tables and scatter plots.
    curve_fit = None


DEFAULT_INPUT_DIR = BASE_DIR / "input"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_BLOCK_SIZE = 25
DEFAULT_DIAGONAL_ANGLE_DEG = 30.0
DEFAULT_OUTLIER_SIGMA = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze steering experiment CSV files from input/ and write results to output/."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--outlier-sigma", type=float, default=DEFAULT_OUTLIER_SIGMA)
    parser.add_argument("--diagonal-angle-deg", type=float, default=DEFAULT_DIAGONAL_ANGLE_DEG)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the current output directory contents before writing new results.",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: Path, clean_output: bool) -> tuple[Path, Path]:
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    return text.strip("_") or "unknown"


def first_non_empty(series: pd.Series) -> str | None:
    non_empty = series.dropna().astype(str).str.strip()
    non_empty = non_empty[non_empty != ""]
    if non_empty.empty:
        return None
    return non_empty.iloc[0]


def normalize_participant_id(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    id_match = re.search(r"ID\s*0*(\d+)", text, flags=re.IGNORECASE)
    if id_match:
        return f"ID{int(id_match.group(1)):03d}"
    if re.fullmatch(r"\d+(\.0)?", text):
        return f"ID{int(float(text)):03d}"
    return text


def infer_participant(path: Path, df: pd.DataFrame) -> str:
    for column in ["participant", "participantId", "participant_id", "subject", "subjectId"]:
        if column in df.columns:
            participant = normalize_participant_id(first_non_empty(df[column]))
            if participant:
                return participant

    for candidate in [path.parent.name, path.stem]:
        participant = normalize_participant_id(candidate)
        if participant and participant != candidate:
            return participant

    filename_match = re.search(r"(?:main|steering)_0*(\d+)_", path.name, flags=re.IGNORECASE)
    if filename_match:
        return f"ID{int(filename_match.group(1)):03d}"

    return path.stem


def infer_condition(path: Path, df: pd.DataFrame) -> str:
    if "conditionId" in df.columns:
        condition = first_non_empty(df["conditionId"])
        if condition:
            return condition
    condition_match = re.search(r"_C(\d+)_", path.name, flags=re.IGNORECASE)
    if condition_match:
        return f"C{condition_match.group(1)}"
    return "unknown"


def discover_csv_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.rglob("*.csv") if path.is_file())


def load_input_frames(input_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, object]] = []

    for csv_path in discover_csv_files(input_dir):
        df = pd.read_csv(csv_path)
        participant = infer_participant(csv_path, df)
        condition = infer_condition(csv_path, df)
        relative_path = csv_path.relative_to(input_dir)
        key = f"{participant}_{csv_path.stem}"

        df = df.copy()
        df["participant"] = participant
        df["source_file"] = str(relative_path)
        if "conditionId" not in df.columns or df["conditionId"].dropna().empty:
            df["conditionId"] = condition

        frames[key] = df
        summaries.append(
            {
                "name": key,
                "participant": participant,
                "conditionId": condition,
                "source_file": str(relative_path),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": "|".join(map(str, df.columns)),
            }
        )

    return frames, pd.DataFrame(summaries)


def bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default)
    values = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "nan", "none"}
    return values.map(lambda value: True if value in true_values else False if value in false_values else default)


def ensure_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "trialInCondition" not in df.columns:
        df["trialInCondition"] = np.arange(1, len(df) + 1)
    df["trialInCondition"] = pd.to_numeric(df["trialInCondition"], errors="coerce")
    if df["trialInCondition"].isna().any():
        fallback = pd.Series(np.arange(1, len(df) + 1), index=df.index)
        df["trialInCondition"] = df["trialInCondition"].fillna(fallback)

    for column in ["success", "countedAsTrial", "deviated"]:
        if column not in df.columns:
            df[column] = False

    for column in ["mtMs", "coreMtMs", "coreEntryMs", "coreExitMs"]:
        if column not in df.columns:
            df[column] = np.nan

    if "trajectory" not in df.columns:
        df["trajectory"] = np.nan

    if "errorType" not in df.columns:
        df["errorType"] = "unknown"

    return df


def build_practice_df(
    raw_frames: dict[str, pd.DataFrame], outlier_sigma: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    practice_list: list[pd.DataFrame] = []
    excluded_list: list[pd.DataFrame] = []

    for _, df in raw_frames.items():
        tmp = ensure_analysis_columns(df)

        if "phase" in tmp.columns:
            tmp = tmp[tmp["phase"].astype(str).str.lower() == "practice"].copy()

        tmp = tmp[bool_series(tmp["countedAsTrial"], default=False)].copy()

        if "coreMtMs" in tmp.columns and pd.to_numeric(tmp["coreMtMs"], errors="coerce").notna().sum() > 0:
            tmp["analysis_mt"] = pd.to_numeric(tmp["coreMtMs"], errors="coerce")
        else:
            tmp["analysis_mt"] = pd.to_numeric(tmp["mtMs"], errors="coerce")

        tmp = tmp[tmp["analysis_mt"].notna()].copy()
        tmp = tmp[bool_series(tmp["success"], default=False)].copy()

        if tmp.empty:
            continue

        mean_mt = tmp["analysis_mt"].mean()
        sd_mt = tmp["analysis_mt"].std()
        lower_threshold = mean_mt - outlier_sigma * sd_mt
        upper_threshold = mean_mt + outlier_sigma * sd_mt

        tmp["mt_mean"] = mean_mt
        tmp["mt_sd"] = sd_mt
        tmp["lower_threshold"] = lower_threshold
        tmp["upper_threshold"] = upper_threshold
        tmp["is_mt_outlier"] = (tmp["analysis_mt"] < lower_threshold) | (
            tmp["analysis_mt"] > upper_threshold
        )

        excluded_list.append(tmp[tmp["is_mt_outlier"]].copy())
        practice_list.append(tmp[~tmp["is_mt_outlier"]].copy())

    practice_df = pd.concat(practice_list, ignore_index=True) if practice_list else pd.DataFrame()
    excluded_trials = pd.concat(excluded_list, ignore_index=True) if excluded_list else pd.DataFrame()

    if not practice_df.empty:
        practice_df = practice_df.sort_values(["participant", "trialInCondition"])

    return practice_df, excluded_trials


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def normalize_trajectory_point(point: object, fallback_t: float) -> dict[str, float] | None:
    if isinstance(point, dict):
        x = to_float(point.get("x"))
        y = to_float(point.get("y"))
        t = to_float(
            point.get("t", point.get("time", point.get("timestamp", point.get("elapsedMs", fallback_t))))
        )
    elif isinstance(point, (list, tuple)) and len(point) >= 2:
        x = to_float(point[0])
        y = to_float(point[1])
        t = to_float(point[2]) if len(point) >= 3 else fallback_t
    else:
        return None

    if not np.isfinite(x) or not np.isfinite(y):
        return None
    if not np.isfinite(t):
        t = fallback_t
    return {"x": x, "y": y, "t": t}


def parse_trajectory(value: object) -> list[dict[str, float]]:
    if isinstance(value, list):
        raw_points = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            raw_points = json.loads(text)
        except json.JSONDecodeError:
            try:
                raw_points = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return []
    else:
        return []

    if isinstance(raw_points, dict):
        raw_points = raw_points.get("points", raw_points.get("trajectory", []))
    if not isinstance(raw_points, list):
        return []

    points: list[dict[str, float]] = []
    for index, point in enumerate(raw_points):
        normalized = normalize_trajectory_point(point, float(index))
        if normalized is not None:
            points.append(normalized)
    return points


def trajectory_arrays(traj: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.array([point["x"] for point in traj], dtype=float)
    ys = np.array([point["y"] for point in traj], dtype=float)
    ts = np.array([point["t"] for point in traj], dtype=float)
    ok = ~(np.isnan(xs) | np.isnan(ys) | np.isnan(ts))
    return xs[ok], ys[ok], ts[ok]


def extract_core_trajectory(
    traj: list[dict[str, float]], core_entry_ms: object, core_exit_ms: object
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    xs, ys, ts = trajectory_arrays(traj)
    if len(xs) < 2:
        return None, None, None

    entry = to_float(core_entry_ms)
    exit_ = to_float(core_exit_ms)
    if not np.isfinite(entry) or not np.isfinite(exit_) or exit_ <= entry:
        return None, None, None

    masks = [
        (ts >= entry) & (ts <= exit_),
        ((ts - ts[0]) >= entry) & ((ts - ts[0]) <= exit_),
    ]
    for mask in masks:
        if mask.sum() >= 2:
            return xs[mask], ys[mask], ts[mask]

    return None, None, None


def calc_tpe_from_trial_flexible(
    row: pd.Series, diagonal_angle_deg: float, prefer_core: bool = True
) -> pd.Series:
    traj = parse_trajectory(row.get("trajectory"))

    if len(traj) < 2:
        return pd.Series(
            {
                "Ae": np.nan,
                "We": np.nan,
                "IDe": np.nan,
                "TPe": np.nan,
                "tpe_valid": False,
                "tpe_method": "invalid",
            }
        )

    core_entry_ms = to_float(row.get("coreEntryMs"))
    core_exit_ms = to_float(row.get("coreExitMs"))
    core_mt_ms = to_float(row.get("coreMtMs"))
    has_core = np.isfinite(core_entry_ms) and np.isfinite(core_exit_ms) and np.isfinite(core_mt_ms)

    if prefer_core and has_core:
        xs, ys, _ = extract_core_trajectory(traj, core_entry_ms, core_exit_ms)
        mt_ms = core_mt_ms
        method = "core"
    else:
        xs, ys, _ = trajectory_arrays(traj)
        mt_ms = to_float(row.get("mtMs", row.get("analysis_mt")))
        method = "full_trajectory"

    if xs is None or ys is None or len(xs) < 2 or not np.isfinite(mt_ms) or mt_ms <= 0:
        return pd.Series(
            {
                "Ae": np.nan,
                "We": np.nan,
                "IDe": np.nan,
                "TPe": np.nan,
                "tpe_valid": False,
                "tpe_method": "invalid",
            }
        )

    dx = np.diff(xs)
    dy = np.diff(ys)
    ae = np.sum(np.sqrt(dx**2 + dy**2))

    theta = np.deg2rad(diagonal_angle_deg)
    dir_x = np.cos(theta)
    dir_y = -np.sin(theta)
    normal_x = -dir_y
    normal_y = dir_x
    perpendicular_positions = xs * normal_x + ys * normal_y
    sigma = np.std(perpendicular_positions, ddof=1)
    we = 4.133 * sigma

    if not np.isfinite(we) or we <= 0:
        return pd.Series(
            {
                "Ae": ae,
                "We": we,
                "IDe": np.nan,
                "TPe": np.nan,
                "tpe_valid": False,
                "tpe_method": method,
            }
        )

    ide = ae / we
    tpe = ide / (mt_ms / 1000)

    return pd.Series(
        {
            "Ae": ae,
            "We": we,
            "IDe": ide,
            "TPe": tpe,
            "tpe_valid": True,
            "tpe_method": method,
        }
    )


def learning_curve(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * (n ** (-b)) + c


def fit_learning_curve(
    x: Iterable[float], y: Iterable[float]
) -> tuple[float, float, float, float, np.ndarray | None, np.ndarray | None]:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
    x = x[ok]
    y = y[ok]

    if len(x) < 4:
        return np.nan, np.nan, np.nan, np.nan, None, None

    if np.nanstd(y) < 1e-12:
        a, b, c = 0.0, 0.0, float(np.nanmean(y))
        y_pred = learning_curve(x, a, b, c)
        return a, b, c, np.nan, x, y_pred

    if curve_fit is None:
        return np.nan, np.nan, np.nan, np.nan, None, None

    c0 = max(0, np.nanmin(y) * 0.8)
    a0 = max(1e-6, np.nanmax(y) - c0)
    b0 = 0.3

    try:
        popt, _ = curve_fit(
            learning_curve,
            x,
            y,
            p0=[a0, b0, c0],
            bounds=([0, 0, 0], [np.inf, 5, np.inf]),
            maxfev=20000,
        )

        a, b, c = popt
        y_pred = learning_curve(x, a, b, c)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = learning_curve(x_fit, a, b, c)
        return a, b, c, r2, x_fit, y_fit
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, None, None


def plot_mt_by_participant(practice_df: pd.DataFrame, figures_dir: Path) -> None:
    for participant_id, group in practice_df.groupby("participant"):
        plt.figure(figsize=(8, 4))
        plt.plot(group["trialInCondition"], group["analysis_mt"], marker="o", markersize=2, linewidth=1)
        condition = group["conditionId"].iloc[0]
        plt.xlabel("Trial number")
        plt.ylabel("coreMtMs or mtMs [ms]")
        plt.title(f"{participant_id} / {condition}")
        plt.grid(True)
        save_fig(figures_dir / f"mt_by_trial_{safe_name(participant_id)}.png")

    plt.figure(figsize=(9, 5))
    for participant_id, group in practice_df.groupby("participant"):
        condition = group["conditionId"].iloc[0]
        plt.plot(
            group["trialInCondition"],
            group["analysis_mt"],
            marker="o",
            markersize=2,
            linewidth=1,
            label=f"{participant_id} / {condition}",
        )
    plt.xlabel("Trial number")
    plt.ylabel("coreMtMs or mtMs [ms]")
    plt.title("Repeated trials by participant")
    plt.legend()
    plt.grid(True)
    save_fig(figures_dir / "mt_by_trial_all_participants.png")


def make_mt_outputs(
    practice_df: pd.DataFrame, block_size: int, figures_dir: Path, tables_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_df = (
        practice_df.groupby("trialInCondition")
        .agg(mean_mt=("analysis_mt", "mean"), sd_mt=("analysis_mt", "std"), n=("analysis_mt", "count"))
        .reset_index()
    )
    save_df(mean_df, tables_dir / "mean_mt_by_trial.csv")

    plt.figure(figsize=(9, 5))
    plt.plot(
        mean_df["trialInCondition"],
        mean_df["mean_mt"],
        marker="o",
        markersize=2,
        linewidth=1.5,
        label="Mean",
    )
    plt.xlabel("Trial number")
    plt.ylabel("Mean coreMtMs or mtMs [ms]")
    plt.title("Mean repeated-trial performance")
    plt.grid(True)
    save_fig(figures_dir / "mean_mt_by_trial.png")

    block_df = practice_df.copy()
    block_df["block25"] = ((block_df["trialInCondition"] - 1) // block_size) + 1
    participant_block_mean_df = (
        block_df.groupby(["participant", "conditionId", "block25"])
        .agg(
            trial_number=("trialInCondition", "max"),
            n=("analysis_mt", "size"),
            mean_mt=("analysis_mt", "mean"),
            sd_mt=("analysis_mt", "std"),
        )
        .reset_index()
    )
    overall_block_mean_df = (
        participant_block_mean_df.groupby("block25")
        .agg(
            trial_number=("trial_number", "max"),
            mean_mt=("mean_mt", "mean"),
            sd_mt=("mean_mt", "std"),
            n=("participant", "count"),
        )
        .reset_index()
    )
    save_df(participant_block_mean_df, tables_dir / "participant_mean_mt_by_25_trial_block.csv")
    save_df(overall_block_mean_df, tables_dir / "overall_mean_mt_by_25_trial_block.csv")

    plt.figure(figsize=(9, 5))
    for participant_id, group in participant_block_mean_df.groupby("participant"):
        condition = group["conditionId"].iloc[0]
        plt.plot(
            group["trial_number"],
            group["mean_mt"],
            marker="o",
            linewidth=1,
            label=f"{participant_id} / {condition}",
        )
    plt.plot(
        overall_block_mean_df["trial_number"],
        overall_block_mean_df["mean_mt"],
        marker="o",
        linewidth=3,
        color="red",
        label="Mean",
    )
    plt.xlabel("Trial number")
    plt.ylabel("Mean coreMtMs or mtMs [ms]")
    plt.title("Repeated trials: 25-trial block mean")
    plt.legend()
    plt.grid(True)
    save_fig(figures_dir / "mt_25_trial_block_mean_by_participant.png")

    return mean_df, participant_block_mean_df, overall_block_mean_df


def make_full_mt_outputs(
    practice_df: pd.DataFrame, block_size: int, figures_dir: Path, tables_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_mt_df = practice_df.copy()
    full_mt_df["full_mt"] = pd.to_numeric(full_mt_df["mtMs"], errors="coerce")
    full_mt_df = full_mt_df[full_mt_df["full_mt"].notna()].copy()
    if full_mt_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    full_mt_df["block25"] = ((full_mt_df["trialInCondition"] - 1) // block_size) + 1
    full_mt_block = (
        full_mt_df.groupby(["participant", "conditionId", "block25"])
        .agg(
            trial_number=("trialInCondition", "max"),
            n=("full_mt", "size"),
            mean_full_mt=("full_mt", "mean"),
            sd_full_mt=("full_mt", "std"),
        )
        .reset_index()
    )
    mean_full_mt_block = (
        full_mt_block.groupby("block25")
        .agg(
            trial_number=("trial_number", "max"),
            mean_full_mt=("mean_full_mt", "mean"),
            sd_full_mt=("mean_full_mt", "std"),
            n=("participant", "count"),
        )
        .reset_index()
    )
    save_df(full_mt_block, tables_dir / "full_mt_by_block.csv")
    save_df(mean_full_mt_block, tables_dir / "mean_full_mt_by_block.csv")

    plt.figure(figsize=(9, 5))
    for participant_id, group in full_mt_block.groupby("participant"):
        condition = group["conditionId"].iloc[0]
        plt.plot(
            group["trial_number"],
            group["mean_full_mt"],
            marker="o",
            linewidth=1.5,
            label=f"{participant_id} / {condition}",
        )
    plt.plot(
        mean_full_mt_block["trial_number"],
        mean_full_mt_block["mean_full_mt"],
        marker="o",
        linewidth=3,
        color="red",
        label="Overall mean",
    )
    plt.xlabel("Trial number")
    plt.ylabel("Mean full MT [ms]")
    plt.title("Mean full MT [ms] by participant with overall mean")
    plt.legend()
    plt.grid(True)
    save_fig(figures_dir / "mean_full_mt_by_block_all_participants.png")

    return full_mt_block, mean_full_mt_block


def make_segment_mt_outputs(
    practice_df: pd.DataFrame, block_size: int, figures_dir: Path, tables_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    segment_df = practice_df.copy()
    for column in ["mtMs", "coreEntryMs", "coreExitMs", "coreMtMs"]:
        segment_df[column] = pd.to_numeric(segment_df[column], errors="coerce")

    valid_mask = (
        segment_df[["mtMs", "coreEntryMs", "coreExitMs", "coreMtMs"]].notna().all(axis=1)
        & (segment_df["coreEntryMs"] >= 0)
        & (segment_df["coreExitMs"] >= segment_df["coreEntryMs"])
        & (segment_df["mtMs"] >= segment_df["coreExitMs"])
        & (segment_df["coreMtMs"] > 0)
    )
    segment_df = segment_df[valid_mask].copy()
    if segment_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    segment_df["pre_core_mt"] = segment_df["coreEntryMs"]
    segment_df["core_mt"] = segment_df["coreMtMs"]
    segment_df["post_core_mt"] = segment_df["mtMs"] - segment_df["coreExitMs"]
    segment_df["block25"] = ((segment_df["trialInCondition"] - 1) // block_size) + 1

    metrics = [
        ("pre_core_mt", "pre-core MT", "Mean pre-core MT [ms]", "mean_pre_core_mt_by_block_all_participants.png"),
        ("core_mt", "core MT", "Mean core MT [ms]", "mean_core_mt_by_block_all_participants.png"),
        ("post_core_mt", "post-core MT", "Mean post-core MT [ms]", "mean_post_core_mt_by_block_all_participants.png"),
    ]
    block_tables: list[pd.DataFrame] = []
    mean_tables: list[pd.DataFrame] = []

    for value_column, metric_label, ylabel, filename in metrics:
        block_df = (
            segment_df.groupby(["participant", "conditionId", "block25"])
            .agg(
                trial_number=("trialInCondition", "max"),
                n=(value_column, "size"),
                mean_mt=(value_column, "mean"),
                sd_mt=(value_column, "std"),
            )
            .reset_index()
        )
        block_df["metric"] = metric_label
        mean_df = (
            block_df.groupby("block25")
            .agg(
                trial_number=("trial_number", "max"),
                mean_mt=("mean_mt", "mean"),
                sd_mt=("mean_mt", "std"),
                n=("participant", "count"),
            )
            .reset_index()
        )
        mean_df["metric"] = metric_label
        block_tables.append(block_df)
        mean_tables.append(mean_df)

        plt.figure(figsize=(9, 5))
        for participant_id, group in block_df.groupby("participant"):
            condition = group["conditionId"].iloc[0]
            plt.plot(
                group["trial_number"],
                group["mean_mt"],
                marker="o",
                linewidth=1.5,
                label=f"{participant_id} / {condition}",
            )
        plt.plot(
            mean_df["trial_number"],
            mean_df["mean_mt"],
            marker="o",
            linewidth=3,
            color="red",
            label="Overall mean",
        )
        plt.xlabel("Trial number")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by participant with overall mean")
        plt.legend()
        plt.grid(True)
        save_fig(figures_dir / filename)

    segment_block_df = pd.concat(block_tables, ignore_index=True)
    mean_segment_block_df = pd.concat(mean_tables, ignore_index=True)
    save_df(segment_block_df, tables_dir / "segment_mt_by_block.csv")
    save_df(mean_segment_block_df, tables_dir / "mean_segment_mt_by_block.csv")

    return segment_block_df, mean_segment_block_df


def make_error_outputs(
    practice_df: pd.DataFrame, block_size: int, figures_dir: Path, tables_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    practice_df = practice_df.copy()
    practice_df["block25"] = ((practice_df["trialInCondition"] - 1) // block_size) + 1
    practice_df["deviation_error"] = bool_series(practice_df["deviated"], default=False).astype(int)

    error_rate_25 = (
        practice_df.groupby(["participant", "conditionId", "block25"])
        .agg(
            trial_number=("trialInCondition", "max"),
            n=("deviation_error", "size"),
            error_count=("deviation_error", "sum"),
            error_rate=("deviation_error", "mean"),
        )
        .reset_index()
    )
    error_rate_25["error_rate_percent"] = error_rate_25["error_rate"] * 100
    save_df(error_rate_25, tables_dir / "error_rate_by_block.csv")

    plt.figure(figsize=(9, 5))
    for participant_id, group in error_rate_25.groupby("participant"):
        condition = group["conditionId"].iloc[0]
        plt.plot(
            group["trial_number"],
            group["error_rate_percent"],
            marker="o",
            linewidth=1.5,
            label=f"{participant_id} / {condition}",
        )
    mean_error_rate_25 = (
        error_rate_25.groupby("block25")
        .agg(
            trial_number=("trial_number", "max"),
            mean_error_rate_percent=("error_rate_percent", "mean"),
            sd_error_rate_percent=("error_rate_percent", "std"),
            n=("participant", "count"),
        )
        .reset_index()
    )
    plt.plot(
        mean_error_rate_25["trial_number"],
        mean_error_rate_25["mean_error_rate_percent"],
        marker="o",
        linewidth=3,
        color="red",
        label="Mean",
    )
    plt.xlabel("Trial number")
    plt.ylabel("Deviation error rate [%]")
    plt.title(f"Deviation error rate by {block_size}-trial block")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    save_fig(figures_dir / "deviation_error_rate_by_block_all_participants.png")

    plt.figure(figsize=(9, 5))
    for participant_id, group in error_rate_25.groupby("participant"):
        condition = group["conditionId"].iloc[0]
        plt.plot(
            group["trial_number"],
            group["error_rate_percent"],
            marker="o",
            linewidth=1.5,
            label=f"{participant_id} / {condition}",
        )
    plt.plot(
        mean_error_rate_25["trial_number"],
        mean_error_rate_25["mean_error_rate_percent"],
        marker="o",
        linewidth=3,
        color="red",
        label="Overall mean",
    )
    plt.xlabel("Trial number")
    plt.ylabel("Deviation error rate [%]")
    plt.title("Deviation error rate by participant with overall mean")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    save_fig(figures_dir / "mean_error_rate_by_block_all_participants.png")

    for participant_id, group in error_rate_25.groupby("participant"):
        condition = group["conditionId"].iloc[0]
        plt.figure(figsize=(8, 4))
        plt.plot(group["trial_number"], group["error_rate_percent"], marker="o", linewidth=1.5)
        plt.xlabel("Trial number")
        plt.ylabel("Deviation error rate [%]")
        plt.title(f"{participant_id} / {condition}: Deviation error rate")
        plt.ylim(0, 100)
        plt.grid(True)
        save_fig(figures_dir / f"deviation_error_rate_{safe_name(participant_id)}.png")

    save_df(mean_error_rate_25, tables_dir / "mean_error_rate_by_block.csv")

    plt.figure(figsize=(9, 5))
    plt.plot(
        mean_error_rate_25["trial_number"],
        mean_error_rate_25["mean_error_rate_percent"],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Trial number")
    plt.ylabel("Mean deviation error rate [%]")
    plt.title(f"Mean deviation error rate by {block_size}-trial block")
    plt.ylim(0, 100)
    plt.grid(True)
    save_fig(figures_dir / "mean_deviation_error_rate_by_block.png")

    return error_rate_25, mean_error_rate_25


def make_tpe_outputs(
    practice_df: pd.DataFrame,
    block_size: int,
    diagonal_angle_deg: float,
    figures_dir: Path,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tpe_df = practice_df.copy()
    tpe_values = tpe_df.apply(lambda row: calc_tpe_from_trial_flexible(row, diagonal_angle_deg), axis=1)
    tpe_df = pd.concat([tpe_df.reset_index(drop=True), tpe_values.reset_index(drop=True)], axis=1)
    tpe_df_valid = tpe_df[tpe_df["tpe_valid"]].copy()

    save_df(tpe_df, tables_dir / "tpe_all_trials.csv")
    save_df(tpe_df_valid, tables_dir / "tpe_valid_trials.csv")

    tpe_count_df = (
        tpe_df_valid.groupby(["participant", "tpe_method"]).size().reset_index(name="n")
        if not tpe_df_valid.empty
        else pd.DataFrame(columns=["participant", "tpe_method", "n"])
    )
    save_df(tpe_count_df, tables_dir / "tpe_counts_by_participant_method.csv")

    if tpe_df_valid.empty:
        return tpe_df_valid, pd.DataFrame(), pd.DataFrame()

    for participant_id, group in tpe_df_valid.groupby("participant"):
        condition = group["conditionId"].iloc[0]
        method = ",".join(group["tpe_method"].unique())
        plt.figure(figsize=(8, 4))
        plt.plot(group["trialInCondition"], group["TPe"], marker="o", markersize=2, linewidth=1)
        plt.xlabel("Trial number")
        plt.ylabel("TPe [1/s]")
        plt.title(f"{participant_id} / {condition}: TPe by trial ({method})")
        plt.grid(True)
        save_fig(figures_dir / f"tpe_by_trial_{safe_name(participant_id)}.png")

    tpe_df_valid["block25"] = ((tpe_df_valid["trialInCondition"] - 1) // block_size) + 1
    tpe_block25 = (
        tpe_df_valid.groupby(["participant", "conditionId", "block25"])
        .agg(
            trial_number=("trialInCondition", "max"),
            n=("TPe", "size"),
            mean_TPe=("TPe", "mean"),
            median_TPe=("TPe", "median"),
            sd_TPe=("TPe", "std"),
            mean_Ae=("Ae", "mean"),
            mean_We=("We", "mean"),
            mean_IDe=("IDe", "mean"),
            mean_coreMtMs=("coreMtMs", "mean"),
        )
        .reset_index()
    )
    save_df(tpe_block25, tables_dir / "tpe_by_block.csv")

    mean_tpe_block25 = (
        tpe_block25.groupby("block25")
        .agg(
            trial_number=("trial_number", "max"),
            mean_TPe=("mean_TPe", "mean"),
            sd_TPe=("mean_TPe", "std"),
            n=("participant", "count"),
        )
        .reset_index()
    )
    save_df(mean_tpe_block25, tables_dir / "mean_tpe_by_block.csv")

    plt.figure(figsize=(9, 5))
    for participant_id, group in tpe_block25.groupby("participant"):
        condition = group["conditionId"].iloc[0]
        plt.plot(
            group["trial_number"],
            group["mean_TPe"],
            marker="o",
            linewidth=1.5,
            label=f"{participant_id} / {condition}",
        )
    plt.plot(
        mean_tpe_block25["trial_number"],
        mean_tpe_block25["mean_TPe"],
        marker="o",
        linewidth=3,
        color="red",
        label="Overall mean",
    )
    plt.xlabel("Trial number")
    plt.ylabel("Mean TPe [1/s]")
    plt.title("Mean TPe [1/s] by participant with overall mean")
    plt.legend()
    plt.grid(True)
    save_fig(figures_dir / "mean_tpe_by_block_all_participants.png")

    plt.figure(figsize=(9, 5))
    plt.plot(mean_tpe_block25["trial_number"], mean_tpe_block25["mean_TPe"], marker="o", linewidth=2)
    plt.xlabel("Trial number")
    plt.ylabel("Mean TPe [1/s]")
    plt.title(f"Overall mean TPe [1/s] by {block_size}-trial block")
    plt.grid(True)
    save_fig(figures_dir / "overall_mean_tpe_by_block.png")

    return tpe_df_valid, tpe_block25, mean_tpe_block25


def make_full_tpe_outputs(
    practice_df: pd.DataFrame,
    block_size: int,
    diagonal_angle_deg: float,
    figures_dir: Path,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_tpe_df = practice_df.copy()
    full_tpe_df["mtMs"] = pd.to_numeric(full_tpe_df["mtMs"], errors="coerce")
    tpe_values = full_tpe_df.apply(
        lambda row: calc_tpe_from_trial_flexible(row, diagonal_angle_deg, prefer_core=False),
        axis=1,
    )
    full_tpe_df = pd.concat([full_tpe_df.reset_index(drop=True), tpe_values.reset_index(drop=True)], axis=1)
    full_tpe_valid = full_tpe_df[full_tpe_df["tpe_valid"]].copy()

    save_df(full_tpe_df, tables_dir / "full_tpe_all_trials.csv")
    save_df(full_tpe_valid, tables_dir / "full_tpe_valid_trials.csv")

    if full_tpe_valid.empty:
        return full_tpe_valid, pd.DataFrame(), pd.DataFrame()

    full_tpe_valid["block25"] = ((full_tpe_valid["trialInCondition"] - 1) // block_size) + 1
    full_tpe_block = (
        full_tpe_valid.groupby(["participant", "conditionId", "block25"])
        .agg(
            trial_number=("trialInCondition", "max"),
            n=("TPe", "size"),
            mean_TPe=("TPe", "mean"),
            median_TPe=("TPe", "median"),
            sd_TPe=("TPe", "std"),
            mean_Ae=("Ae", "mean"),
            mean_We=("We", "mean"),
            mean_IDe=("IDe", "mean"),
            mean_full_mt=("mtMs", "mean"),
        )
        .reset_index()
    )
    mean_full_tpe_block = (
        full_tpe_block.groupby("block25")
        .agg(
            trial_number=("trial_number", "max"),
            mean_TPe=("mean_TPe", "mean"),
            sd_TPe=("mean_TPe", "std"),
            n=("participant", "count"),
        )
        .reset_index()
    )
    save_df(full_tpe_block, tables_dir / "full_tpe_by_block.csv")
    save_df(mean_full_tpe_block, tables_dir / "mean_full_tpe_by_block.csv")

    plt.figure(figsize=(9, 5))
    for participant_id, group in full_tpe_block.groupby("participant"):
        condition = group["conditionId"].iloc[0]
        plt.plot(
            group["trial_number"],
            group["mean_TPe"],
            marker="o",
            linewidth=1.5,
            label=f"{participant_id} / {condition}",
        )
    plt.plot(
        mean_full_tpe_block["trial_number"],
        mean_full_tpe_block["mean_TPe"],
        marker="o",
        linewidth=3,
        color="red",
        label="Overall mean",
    )
    plt.xlabel("Trial number")
    plt.ylabel("Mean full-trajectory TPe [1/s]")
    plt.title("Mean full-trajectory TPe [1/s] by participant with overall mean")
    plt.legend()
    plt.grid(True)
    save_fig(figures_dir / "mean_full_tpe_by_block_all_participants.png")

    return full_tpe_valid, full_tpe_block, mean_full_tpe_block


def make_hundred_trial_window_outputs(
    practice_df: pd.DataFrame,
    tpe_df_valid: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []

    mt_df = practice_df.copy()
    mt_df["value"] = pd.to_numeric(mt_df["analysis_mt"], errors="coerce")
    mt_df["metric"] = "core_mt"
    mt_df["metric_label"] = "Core MT (ID001: full MT)"
    mt_df["ylabel"] = "Mean core MT [ms]"
    mt_df["filename_prefix"] = "core_mt"
    metric_frames.append(
        mt_df[["participant", "conditionId", "trialInCondition", "value", "metric", "metric_label", "ylabel", "filename_prefix"]]
    )

    error_df = practice_df.copy()
    error_df["value"] = bool_series(error_df["deviated"], default=False).astype(float) * 100
    error_df["metric"] = "error_rate"
    error_df["metric_label"] = "Error rate"
    error_df["ylabel"] = "Error rate [%]"
    error_df["filename_prefix"] = "error_rate"
    metric_frames.append(
        error_df[
            ["participant", "conditionId", "trialInCondition", "value", "metric", "metric_label", "ylabel", "filename_prefix"]
        ]
    )

    throughput_df = tpe_df_valid.copy()
    throughput_df["value"] = pd.to_numeric(throughput_df["TPe"], errors="coerce")
    throughput_df["metric"] = "throughput"
    throughput_df["metric_label"] = "Throughput"
    throughput_df["ylabel"] = "Mean throughput TPe [1/s]"
    throughput_df["filename_prefix"] = "throughput"
    metric_frames.append(
        throughput_df[
            ["participant", "conditionId", "trialInCondition", "value", "metric", "metric_label", "ylabel", "filename_prefix"]
        ]
    )

    plot_source = pd.concat(metric_frames, ignore_index=True)
    plot_source["trialInCondition"] = pd.to_numeric(plot_source["trialInCondition"], errors="coerce")
    plot_source = plot_source[plot_source["trialInCondition"].notna() & plot_source["value"].notna()].copy()
    if plot_source.empty:
        return pd.DataFrame(), pd.DataFrame()

    windows = [(1, 100), (101, 200), (201, 300), (301, 400)]
    participant_tables: list[pd.DataFrame] = []
    overall_tables: list[pd.DataFrame] = []
    axis_limit_rows: list[dict[str, object]] = []

    for metric, metric_df in plot_source.groupby("metric", sort=False):
        metric_label = metric_df["metric_label"].iloc[0]
        ylabel = metric_df["ylabel"].iloc[0]
        filename_prefix = metric_df["filename_prefix"].iloc[0]
        metric_window_summaries: list[tuple[int, int, str, pd.DataFrame, pd.DataFrame]] = []
        y_values: list[pd.Series] = []

        for start, end in windows:
            window_df = metric_df[
                (metric_df["trialInCondition"] >= start) & (metric_df["trialInCondition"] <= end)
            ].copy()
            if window_df.empty:
                continue

            if metric == "error_rate":
                window_df["plot_index"] = ((window_df["trialInCondition"] - start) // 10).astype(int) + 1
                aggregation = "10-trial mean"
                title_measure = "10-trial means"
            else:
                window_df["plot_index"] = window_df["trialInCondition"].astype(int)
                aggregation = "single trial"
                title_measure = "single-trial values"
            window_df["window_start"] = start
            window_df["window_end"] = end
            window_df["window_label"] = f"{start}-{end}"

            participant_summary = (
                window_df.groupby(
                    [
                        "metric",
                        "metric_label",
                        "window_start",
                        "window_end",
                        "window_label",
                        "participant",
                        "conditionId",
                        "plot_index",
                    ]
                )
                .agg(
                    trial_number=("trialInCondition", "max"),
                    n=("value", "size"),
                    mean_value=("value", "mean"),
                    sd_value=("value", "std"),
                )
                .reset_index()
            )
            participant_summary["aggregation"] = aggregation
            participant_summary["title_measure"] = title_measure
            overall_summary = (
                participant_summary.groupby(
                    ["metric", "metric_label", "window_start", "window_end", "window_label", "plot_index"]
                )
                .agg(
                    trial_number=("trial_number", "max"),
                    n=("participant", "count"),
                    mean_value=("mean_value", "mean"),
                    sd_value=("mean_value", "std"),
                )
                .reset_index()
            )
            overall_summary["aggregation"] = aggregation
            overall_summary["title_measure"] = title_measure
            participant_tables.append(participant_summary)
            overall_tables.append(overall_summary)
            metric_window_summaries.append((start, end, title_measure, participant_summary, overall_summary))
            y_values.extend([participant_summary["mean_value"], overall_summary["mean_value"]])

        if not metric_window_summaries:
            continue

        finite_values = pd.concat(y_values, ignore_index=True)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.empty:
            continue

        y_min = float(finite_values.min())
        y_max = float(finite_values.max())
        y_span = y_max - y_min
        y_pad = y_span * 0.05 if y_span > 0 else max(abs(y_max) * 0.05, 1.0)
        y_lower = y_min - y_pad
        y_upper = y_max + y_pad
        if metric == "error_rate":
            y_lower = max(0.0, y_lower)
            y_upper = min(100.0, y_upper)
        axis_limit_rows.append(
            {
                "metric": metric,
                "metric_label": metric_label,
                "data_min": y_min,
                "data_max": y_max,
                "axis_min": y_lower,
                "axis_max": y_upper,
            }
        )

        for start, end, title_measure, participant_summary, overall_summary in metric_window_summaries:
            plt.figure(figsize=(9, 5))
            for participant_id, group in participant_summary.groupby("participant"):
                condition = group["conditionId"].iloc[0]
                plt.plot(
                    group["trial_number"],
                    group["mean_value"],
                    marker="o",
                    markersize=2 if metric != "error_rate" else 6,
                    linewidth=1 if metric != "error_rate" else 1.5,
                    label=f"{participant_id} / {condition}",
                )
            plt.plot(
                overall_summary["trial_number"],
                overall_summary["mean_value"],
                marker="o",
                markersize=3 if metric != "error_rate" else 6,
                linewidth=2 if metric != "error_rate" else 3,
                color="red",
                label="Overall mean",
            )
            plt.xlabel("Trial number")
            plt.ylabel(ylabel)
            plt.title(f"{metric_label}: {title_measure} in trials {start}-{end}")
            plt.ylim(y_lower, y_upper)
            plt.legend()
            plt.grid(True)
            save_fig(figures_dir / f"{filename_prefix}_trials_{start:03d}_{end:03d}_10trial_mean.png")

    participant_df = pd.concat(participant_tables, ignore_index=True) if participant_tables else pd.DataFrame()
    overall_df = pd.concat(overall_tables, ignore_index=True) if overall_tables else pd.DataFrame()
    save_df(participant_df, tables_dir / "hundred_trial_10trial_summary.csv")
    save_df(overall_df, tables_dir / "hundred_trial_10trial_overall_summary.csv")
    save_df(participant_df, tables_dir / "hundred_trial_window_summary.csv")
    save_df(overall_df, tables_dir / "hundred_trial_window_overall_summary.csv")
    save_df(pd.DataFrame(axis_limit_rows), tables_dir / "hundred_trial_10trial_axis_limits.csv")

    return participant_df, overall_df


def make_hundred_trial_mean_progression_outputs(
    practice_df: pd.DataFrame,
    tpe_df_valid: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []

    mt_df = practice_df.copy()
    mt_df["value"] = pd.to_numeric(mt_df["analysis_mt"], errors="coerce")
    mt_df["metric"] = "mt"
    mt_df["metric_label"] = "MT (ID001: full MT, others: core MT)"
    mt_df["ylabel"] = "Mean MT [ms]"
    mt_df["filename"] = "hundred_trial_mean_mt_progression.png"
    metric_frames.append(
        mt_df[["participant", "conditionId", "trialInCondition", "value", "metric", "metric_label", "ylabel", "filename"]]
    )

    tpe_df = tpe_df_valid.copy()
    tpe_df["value"] = pd.to_numeric(tpe_df["TPe"], errors="coerce")
    tpe_df["metric"] = "throughput"
    tpe_df["metric_label"] = "Throughput TPe"
    tpe_df["ylabel"] = "Mean TPe [1/s]"
    tpe_df["filename"] = "hundred_trial_mean_tpe_progression.png"
    metric_frames.append(
        tpe_df[["participant", "conditionId", "trialInCondition", "value", "metric", "metric_label", "ylabel", "filename"]]
    )

    plot_source = pd.concat(metric_frames, ignore_index=True)
    plot_source["trialInCondition"] = pd.to_numeric(plot_source["trialInCondition"], errors="coerce")
    plot_source = plot_source[plot_source["trialInCondition"].notna() & plot_source["value"].notna()].copy()
    if plot_source.empty:
        return pd.DataFrame(), pd.DataFrame()

    plot_source["hundred_trial_block"] = ((plot_source["trialInCondition"] - 1) // 100).astype(int) + 1
    plot_source = plot_source[plot_source["hundred_trial_block"].between(1, 4)].copy()
    plot_source["window_start"] = (plot_source["hundred_trial_block"] - 1) * 100 + 1
    plot_source["window_end"] = plot_source["hundred_trial_block"] * 100
    plot_source["window_label"] = plot_source["window_start"].astype(str) + "-" + plot_source["window_end"].astype(str)
    if plot_source.empty:
        return pd.DataFrame(), pd.DataFrame()

    participant_summary = (
        plot_source.groupby(
            [
                "metric",
                "metric_label",
                "ylabel",
                "filename",
                "hundred_trial_block",
                "window_start",
                "window_end",
                "window_label",
                "participant",
                "conditionId",
            ]
        )
        .agg(
            trial_number=("trialInCondition", "max"),
            n=("value", "size"),
            mean_value=("value", "mean"),
            sd_value=("value", "std"),
        )
        .reset_index()
    )
    participant_summary["aggregation"] = "100-trial mean"

    overall_summary = (
        participant_summary.groupby(
            ["metric", "metric_label", "ylabel", "filename", "hundred_trial_block", "window_start", "window_end", "window_label"]
        )
        .agg(
            trial_number=("trial_number", "max"),
            n=("participant", "count"),
            mean_value=("mean_value", "mean"),
            sd_value=("mean_value", "std"),
        )
        .reset_index()
    )
    overall_summary["aggregation"] = "100-trial mean"

    for metric, metric_summary in participant_summary.groupby("metric", sort=False):
        metric_overall = overall_summary[overall_summary["metric"] == metric]
        metric_label = metric_summary["metric_label"].iloc[0]
        ylabel = metric_summary["ylabel"].iloc[0]
        filename = metric_summary["filename"].iloc[0]

        y_values = pd.concat([metric_summary["mean_value"], metric_overall["mean_value"]], ignore_index=True)
        y_values = y_values[np.isfinite(y_values)]
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        y_span = y_max - y_min
        y_pad = y_span * 0.05 if y_span > 0 else max(abs(y_max) * 0.05, 1.0)

        plt.figure(figsize=(9, 5))
        for participant_id, group in metric_summary.groupby("participant"):
            condition = group["conditionId"].iloc[0]
            plt.plot(
                group["window_end"],
                group["mean_value"],
                marker="o",
                linewidth=1.5,
                label=f"{participant_id} / {condition}",
            )
        plt.plot(
            metric_overall["window_end"],
            metric_overall["mean_value"],
            marker="o",
            linewidth=3,
            color="red",
            label="Overall mean",
        )
        plt.xlabel("Trial number")
        plt.ylabel(ylabel)
        plt.title(f"{metric_label}: 100-trial mean progression")
        plt.ylim(y_min - y_pad, y_max + y_pad)
        plt.xticks([100, 200, 300, 400])
        plt.legend()
        plt.grid(True)
        save_fig(figures_dir / filename)

    save_df(participant_summary, tables_dir / "hundred_trial_mean_progression_summary.csv")
    save_df(overall_summary, tables_dir / "hundred_trial_mean_progression_overall_summary.csv")

    return participant_summary, overall_summary


def make_learning_curve_outputs(
    practice_df: pd.DataFrame,
    tpe_df_valid: pd.DataFrame,
    block_size: int,
    figures_dir: Path,
    tables_dir: Path,
) -> pd.DataFrame:
    mt_df = practice_df.copy()
    mt_df["block25"] = ((mt_df["trialInCondition"] - 1) // block_size) + 1
    mt_block = (
        mt_df.groupby(["participant", "block25"])
        .agg(N=("trialInCondition", "max"), y=("analysis_mt", "mean"), conditionId=("conditionId", "first"))
        .reset_index()
    )
    mt_block["metric"] = "MT"

    er_df = practice_df.copy()
    er_df["block25"] = ((er_df["trialInCondition"] - 1) // block_size) + 1
    er_df["deviation_error"] = bool_series(er_df["deviated"], default=False).astype(int)
    er_block = (
        er_df.groupby(["participant", "block25"])
        .agg(N=("trialInCondition", "max"), y=("deviation_error", "mean"), conditionId=("conditionId", "first"))
        .reset_index()
    )
    er_block["metric"] = "ER"

    inv_tpe_df = tpe_df_valid.copy()
    inv_tpe_df = inv_tpe_df[(inv_tpe_df["TPe"].notna()) & (inv_tpe_df["TPe"] > 0)].copy()
    if not inv_tpe_df.empty:
        inv_tpe_df["inv_TPe"] = 1 / inv_tpe_df["TPe"]
        inv_tpe_df["block25"] = ((inv_tpe_df["trialInCondition"] - 1) // block_size) + 1
        inv_tpe_block = (
            inv_tpe_df.groupby(["participant", "block25"])
            .agg(N=("trialInCondition", "max"), y=("inv_TPe", "mean"), conditionId=("conditionId", "first"))
            .reset_index()
        )
        inv_tpe_block["metric"] = "1/TPe"
    else:
        inv_tpe_block = pd.DataFrame(columns=["participant", "block25", "N", "y", "conditionId", "metric"])

    plot_df = pd.concat([mt_block, er_block, inv_tpe_block], ignore_index=True)
    save_df(plot_df, tables_dir / "learning_curve_plot_data.csv")

    participants = sorted(plot_df["participant"].dropna().unique())
    metrics = ["MT", "ER", "1/TPe"]
    fit_results: list[dict[str, object]] = []

    for participant in participants:
        for metric in metrics:
            group = plot_df[(plot_df["participant"] == participant) & (plot_df["metric"] == metric)].copy()
            a, b, c, r2, _, _ = fit_learning_curve(group["N"], group["y"])
            fit_results.append(
                {
                    "participant": participant,
                    "metric": metric,
                    "conditionId": group["conditionId"].iloc[0] if len(group) > 0 else np.nan,
                    "n_blocks": len(group),
                    "a": a,
                    "b": b,
                    "c": c,
                    "r2": r2,
                }
            )

    fit_results_df = pd.DataFrame(fit_results)
    save_df(fit_results_df, tables_dir / "learning_curve_fit_results.csv")

    if len(participants) == 0:
        return fit_results_df

    fig, axes = plt.subplots(
        nrows=len(participants),
        ncols=len(metrics),
        figsize=(15, 4 * len(participants)),
        squeeze=False,
    )

    for row_index, participant in enumerate(participants):
        for column_index, metric in enumerate(metrics):
            ax = axes[row_index, column_index]
            group = plot_df[(plot_df["participant"] == participant) & (plot_df["metric"] == metric)].copy()
            result = fit_results_df[
                (fit_results_df["participant"] == participant) & (fit_results_df["metric"] == metric)
            ].iloc[0]

            ax.scatter(group["N"], group["y"], s=28)
            _, _, _, _, x_fit, y_fit = fit_learning_curve(group["N"], group["y"])
            if x_fit is not None:
                ax.plot(x_fit, y_fit, linewidth=2)

            if metric == "MT":
                ylabel = "MT [ms]"
            elif metric == "ER":
                ylabel = "Error rate"
                ax.set_ylim(0, 1.05)
            else:
                ylabel = "1 / TPe [s]"

            condition = result["conditionId"]
            ax.set_title(
                f"{participant} / {condition} / {metric}\n"
                f"a={result['a']:.3f}, b={result['b']:.3f}, c={result['c']:.3f}, R2={result['r2']:.3f}"
            )
            ax.set_xlabel("Trial number")
            ax.set_ylabel(ylabel)
            ax.grid(True)

    fig.tight_layout()
    fig.savefig(figures_dir / "learning_curve_3x3.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return fit_results_df


def write_summary(
    output_dir: Path,
    raw_summary: pd.DataFrame,
    practice_df: pd.DataFrame,
    excluded_trials: pd.DataFrame,
    tpe_valid_trials: pd.DataFrame,
    fit_results: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "Steering experiment analysis summary",
        "",
        f"input_dir: {args.input_dir}",
        f"output_dir: {args.output_dir}",
        f"block_size: {args.block_size}",
        f"outlier_sigma: {args.outlier_sigma}",
        f"diagonal_angle_deg: {args.diagonal_angle_deg}",
        f"scipy_curve_fit_available: {curve_fit is not None}",
        "",
        f"csv_files: {len(raw_summary)}",
        f"practice_trials_after_filters: {len(practice_df)}",
        f"mt_outlier_trials: {len(excluded_trials)}",
        f"tpe_valid_trials: {len(tpe_valid_trials)}",
        f"learning_curve_fit_rows: {len(fit_results)}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    figures_dir, tables_dir = ensure_output_dir(output_dir, args.clean_output)

    raw_frames, raw_summary = load_input_frames(input_dir)
    save_df(raw_summary, tables_dir / "raw_file_summary.csv")

    if not raw_frames:
        message = (
            "No CSV files were found. Put experiment CSV files in the input/ directory, "
            "then run python3 analysis.py again."
        )
        (output_dir / "summary.txt").write_text(message + "\n", encoding="utf-8")
        print(message)
        return 0

    practice_df, excluded_trials = build_practice_df(raw_frames, args.outlier_sigma)
    save_df(practice_df, tables_dir / "practice_trials_after_filters.csv")
    save_df(excluded_trials, tables_dir / "mt_outlier_excluded_trials.csv")

    if practice_df.empty:
        message = "No analyzable practice trials remained after filtering."
        (output_dir / "summary.txt").write_text(message + "\n", encoding="utf-8")
        print(message)
        return 0

    plot_mt_by_participant(practice_df, figures_dir)
    make_mt_outputs(practice_df, args.block_size, figures_dir, tables_dir)
    make_full_mt_outputs(practice_df, args.block_size, figures_dir, tables_dir)
    make_segment_mt_outputs(practice_df, args.block_size, figures_dir, tables_dir)
    make_error_outputs(practice_df, args.block_size, figures_dir, tables_dir)
    tpe_valid_trials, _, _ = make_tpe_outputs(
        practice_df, args.block_size, args.diagonal_angle_deg, figures_dir, tables_dir
    )
    make_hundred_trial_window_outputs(practice_df, tpe_valid_trials, figures_dir, tables_dir)
    make_hundred_trial_mean_progression_outputs(practice_df, tpe_valid_trials, figures_dir, tables_dir)
    make_full_tpe_outputs(practice_df, args.block_size, args.diagonal_angle_deg, figures_dir, tables_dir)
    fit_results = make_learning_curve_outputs(practice_df, tpe_valid_trials, args.block_size, figures_dir, tables_dir)

    write_summary(output_dir, raw_summary, practice_df, excluded_trials, tpe_valid_trials, fit_results, args)

    print(f"Analysis complete: {output_dir}")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")
    return 0


def main() -> int:
    return run_analysis(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
