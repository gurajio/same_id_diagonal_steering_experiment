"""水平直線ステアリング実験のCSVを分析する．

元コードの記述統計，MT/We/Ae/TPe/ER，学習曲線，条件比較をCore区間で行う．
MTはcoreMtMs，軌跡はcoreEntryMs〜coreExitMs，ERはCore内逸脱・未完了に基づく．
ERの分母はCore進入試行，分子はCore内逸脱またはCore未完了の試行数（重複は1試行）．
Core未進入は除外し，Core通過後の操作結果はCoreの成績に影響させない．
全操作区間のsuccess/errorType/deviatedはCoreの成否判定には使わない．
入力CSVは読み取り専用で扱い，分析結果はoutputに出力する．
--pilot-summary --phase all で反復・事後を分けた平均・不偏分散と推移をpilot_summaryに出力する．
"""

from pathlib import Path
from datetime import datetime
from io import BytesIO
import argparse
import errno
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from scipy.optimize import curve_fit
import statsmodels.formula.api as smf
import statsmodels.api as sm


# ============================================================
# 0. 設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "inputCSV"  # CSVフォルダ
OUT_DIR = BASE_DIR / "output"    # 出力フォルダ
CONDITION_REPORT_DIR = OUT_DIR / "condition_report"

# CSV_FILENAMESが空の場合に使う既定ファイル．両方空ならinputCSV内の全CSVを使う．
DEFAULT_CSV_FILE = "same_id_steering_main_001_C2-C3_20260920_172235.csv"

# 複数ファイルを指定する場合に使う．優先順位はコマンドライン > このリスト > DEFAULT_CSV_FILE．
CSV_FILENAMES = []

# このコードはCore区間専用．全操作時間へのフォールバックはしない．
MT_SOURCE = "coreMtMs"
ANALYSIS_SCOPE = "core"

# 元CSVのcountedAsTrialにかかわらず，Coreに進入した試行をERの分母にする．
CORE_TRIAL_DENOMINATOR = "core_entered"

# 実験の逸脱閾値と同じpx単位．Core内の記録だけで判定する．
CORE_EXCESSIVE_DEVIATION_PX = 24.0

# "learning": practice/pilotのみ．"post": 事後試行のみ．"all": 全フェーズ．
ANALYSIS_PHASE = "learning"

BOOT_N = 5000
BIN_SIZE = 20
BIN_SIZES = [10, 50, 100]
HYPOTHESIS_BIN_SIZE = 20
# 元コードの分析上限を維持する．Nは未完了のやり直しを含むCore進入回数．
MAX_TRIAL_N = 400

# MT/TPeの移動平均
MT_ROLLING_WINDOW = 10
TPE_ROLLING_WINDOW = 20

# We = 4.133 * sigma_d
WE_COEFFICIENT = 4.133

# Core通過完了試行のMT外れ値を削除するかどうか
DELETE_OUTLIER = True
OUTLIER_SIGMA = 3.0

# Core内で逸脱した通過完了試行をMT/TPe計算に含めるかどうか
# False: Core通過完了かつCore内逸脱なしのみ使う
# True : Core通過完了ならCore内の逸脱を含める（下記の逸脱時間割合を適用）
INCLUDE_DEVIATED_TRIALS = True

# 経路はみ出しは基本的に許容するが，試行時間のこの割合以上はみ出していた場合は除外する．
# Core内の逸脱時間 / coreMtMsを使う．全区間のdeviationTotalMsは使わない．
# コマンドライン引数 --deviation-exclusion-ratio で実行時に変更できる．
DEVIATION_EXCLUSION_RATIO = 0.80

# 日本語フォント．MacならHiragino Sans，WindowsならYu Gothicに変更するとよい．
JAPANESE_FONT = "Yu Gothic"

# steering_experiment_config.jsの水平版3条件に対応する．
COND_SPECS = {
    "width_wide": {"number": 1, "A": 540, "W": 30, "ID": 18, "name": "Wide-width"},
    "length_long": {"number": 2, "A": 810, "W": 30, "ID": 27, "name": "Long-distance"},
    "width_narrow": {"number": 3, "A": 540, "W": 20, "ID": 27, "name": "Narrow-width"},
}

COND_DISPLAY = {
    cond: f"{spec['name']} A{spec['A']} W{spec['W']} ID{spec['ID']}"
    for cond, spec in COND_SPECS.items()
}

COND_ORDER = [
    "width_wide",
    "length_long",
    "width_narrow",
]

COND_NUMBER = {cond: spec["number"] for cond, spec in COND_SPECS.items()}

COND_ID_GROUP = {cond: f"ID{spec['ID']}" for cond, spec in COND_SPECS.items()}

SAME_ID_PAIRS = [("ID27", ["length_long", "width_narrow"])]

# 差は左条件 - 右条件．条件の一部だけを収録したCSVにも対応する．
CONTRAST_DEFS = [
    ("same_ID27_length_long_minus_width_narrow", "Same ID27: C2 long - C3 narrow", "length_long", "width_narrow"),
    ("width_wide_minus_width_narrow", "Width manipulation: C1 wide - C3 narrow", "width_wide", "width_narrow"),
    ("length_long_minus_width_wide", "Distance manipulation: C2 long - C1 wide", "length_long", "width_wide"),
]


# ============================================================
# 1. 汎用関数
# ============================================================

def setup_matplotlib():
    candidates = [
        JAPANESE_FONT,
        "Hiragino Sans",
        "Yu Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
        "Noto Sans CJK",
        "IPAexGothic",
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    selected_font = next((font for font in candidates if font in available_fonts), None)

    if selected_font is not None:
        mpl.rcParams["font.family"] = selected_font
    else:
        mpl.rcParams["font.family"] = JAPANESE_FONT

    mpl.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def fmt_value(value):
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):g}"
    except Exception:
        return str(value)


def save_png_figure(fig, filename, **kwargs):
    """PNGを保存し，既存画像の上書きが拒否された場合だけ日時付き別名を使う．"""
    target = Path(filename)
    # 描画処理とファイルを開く処理を分け，描画エラーを上書き失敗と誤認しない．
    with BytesIO() as buffer:
        fig.savefig(buffer, format="png", **kwargs)
        try:
            stream = target.open("wb")
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.EACCES, errno.EPERM, errno.EBUSY} or not target.is_file():
                raise
            # Windowsの画像プレビュー等で既存PNGを上書きできない場合の回避策．
            # 元ファイルは削除せず，CSVの保存先・処理にも影響させない．
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fallback = target.with_name(f"{target.stem}_{stamp}{target.suffix}")
            stream = fallback.open("xb")
            with stream:
                stream.write(buffer.getvalue())
            print(f"[PNG save warning] Could not overwrite {target}: {error}")
            print(f"[PNG saved under another name] {fallback}")
            return fallback
        with stream:
            stream.write(buffer.getvalue())
    return target


def make_condition_metric_label(cond):
    cond = str(cond)
    label = COND_DISPLAY.get(cond, cond)
    number = COND_NUMBER.get(cond)
    return f"{label} / C{number}" if number is not None else label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze MT, ER, and TPe exclusively within the horizontal steering core interval."
    )
    parser.add_argument(
        "csv_filenames",
        nargs="*",
        help="CSV filename(s) in inputCSV, or absolute paths. Defaults: CSV_FILENAMES, then DEFAULT_CSV_FILE, then all CSV files.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ignore CSV_FILENAMES and DEFAULT_CSV_FILE and analyze all CSV files in inputCSV.",
    )
    parser.add_argument(
        "--phase", choices=["learning", "practice", "pilot", "post", "all"],
        default=ANALYSIS_PHASE,
        help="Trial phase to analyze. learning includes practice and pilot, excluding post trials.",
    )
    parser.add_argument(
        "--mt-source", choices=["coreMtMs"], default=MT_SOURCE,
        help="Core-only analysis: MT, trajectories, errors, and deviation exclusions all use the core interval.",
    )
    parser.add_argument(
        "--deviation-exclusion-ratio",
        type=float,
        default=None,
        help=(
            "Override DEVIATION_EXCLUSION_RATIO. "
            "Example: 0.8 excludes trials with core deviation time / coreMtMs >= 0.8."
        ),
    )
    outlier_group = parser.add_mutually_exclusive_group()
    outlier_group.add_argument(
        "--delete-outlier",
        dest="delete_outlier",
        action="store_true",
        default=None,
        help="Remove Core MT outliers from completed Core traversals.",
    )
    outlier_group.add_argument(
        "--keep-outlier",
        dest="delete_outlier",
        action="store_false",
        help="Keep Core MT outliers in completed Core traversals.",
    )
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=None,
        help="Override OUTLIER_SIGMA used for MT outlier detection.",
    )
    deviated_group = parser.add_mutually_exclusive_group()
    deviated_group.add_argument(
        "--include-deviated-trials",
        dest="include_deviated_trials",
        action="store_true",
        default=None,
        help="Include completed Core traversals with Core deviations in MT/TPe calculations.",
    )
    deviated_group.add_argument(
        "--exclude-deviated-trials",
        dest="include_deviated_trials",
        action="store_false",
        help="Exclude Core-deviated traversals from MT/TPe calculations; retain them in Core ER.",
    )
    parser.add_argument(
        "--pilot-summary", action="store_true",
        help="Write mean/variance CSVs and PNGs to output/pilot_summary, separating phases. Use --phase all to include post trials. Skip model fitting.",
    )
    return parser.parse_args()


def apply_cli_overrides(args):
    """
    コード上部の設定値を，必要な場合だけコマンドライン引数で上書きする．
    何も指定しなければ，スクリプト内のデフォルト設定をそのまま使う．
    """
    global DEVIATION_EXCLUSION_RATIO
    global DELETE_OUTLIER
    global OUTLIER_SIGMA
    global INCLUDE_DEVIATED_TRIALS

    if args.deviation_exclusion_ratio is not None:
        if not 0 <= args.deviation_exclusion_ratio <= 1:
            raise ValueError("--deviation-exclusion-ratio must be between 0 and 1.")
        DEVIATION_EXCLUSION_RATIO = args.deviation_exclusion_ratio

    if args.delete_outlier is not None:
        DELETE_OUTLIER = args.delete_outlier

    if args.outlier_sigma is not None:
        if args.outlier_sigma <= 0:
            raise ValueError("--outlier-sigma must be greater than 0.")
        OUTLIER_SIGMA = args.outlier_sigma

    if args.include_deviated_trials is not None:
        INCLUDE_DEVIATED_TRIALS = args.include_deviated_trials


def find_csv_files(csv_filenames=None, read_all=False):
    """
    inputCSV内のCSVファイルを取得する．
    コマンドラインまたはCSV_FILENAMESで指定された場合は，そのファイルだけを対象にする．
    """
    if read_all:
        return sorted(CSV_DIR.glob("*.csv"))

    names = list(csv_filenames or CSV_FILENAMES)
    if not names and DEFAULT_CSV_FILE:
        names = [DEFAULT_CSV_FILE]
    if names:
        csv_files = []
        for name in names:
            csv_path = CSV_DIR / name
            if not csv_path.exists():
                available = "\n".join(f"  - {p.name}" for p in sorted(CSV_DIR.glob("*.csv")))
                raise FileNotFoundError(
                    f"CSV not found: {csv_path}\n"
                    f"Available CSV files:\n{available if available else '  (none)'}"
                )
            if csv_path.suffix.lower() != ".csv":
                raise ValueError(f"Please specify a CSV file: {csv_path.name}")
            csv_files.append(csv_path)
        return csv_files

    return sorted(CSV_DIR.glob("*.csv"))


def pick_col(df, candidates, required=False):
    """
    複数の候補列名から，存在する列を1つ選ぶ．
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(f"Required column not found. Candidates: {candidates}")
    return None


def build_performance_trial_mask(df):
    """
    MT/TPe分析に使う試行を設定から決める．
    ER計算用のerror_countedとは分けて扱う．
    """
    mask = df["core_completed"] & df["is_counted_trial"]
    if not INCLUDE_DEVIATED_TRIALS:
        mask = mask & ~df["core_deviated"]

    if "exclude_by_deviation_ratio" in df.columns:
        mask = mask & ~df["exclude_by_deviation_ratio"].fillna(False)

    mask = mask & np.isfinite(df["MT"]) & (df["MT"] > 0)
    return mask.fillna(False)


def mark_mt_outliers(df, performance_mask):
    """
    MT/TPe分析対象のMTについて，全体平均±OUTLIER_SIGMAσの外れ値を印付けする．
    """
    is_outlier = pd.Series(False, index=df.index)
    if not DELETE_OUTLIER:
        return is_outlier

    mt = pd.to_numeric(df.loc[performance_mask, "MT"], errors="coerce").dropna()
    if len(mt) < 2:
        return is_outlier

    mt_mean = mt.mean()
    mt_sd = mt.std()
    if pd.isna(mt_sd) or mt_sd == 0:
        return is_outlier

    lower = mt_mean - OUTLIER_SIGMA * mt_sd
    upper = mt_mean + OUTLIER_SIGMA * mt_sd
    is_outlier.loc[performance_mask] = (
        (df.loc[performance_mask, "MT"] < lower)
        | (df.loc[performance_mask, "MT"] > upper)
    )
    return is_outlier


def parse_trajectory_json(s):
    if isinstance(s, list):
        return s
    if pd.isna(s):
        return []
    try:
        traj = json.loads(s)
        if isinstance(traj, list):
            return traj
    except Exception:
        return []
    return []


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def read_timed_trajectory(row, trajectory_col):
    """CSVの時刻付き軌跡を読み，必要なら最後の離脱座標を補間用に加える．"""
    points = []
    for point in parse_trajectory_json(row[trajectory_col]):
        try:
            x, y = float(point["x"]), float(point["y"])
            t = float(point["t"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite([x, y, t]).all() and t >= 0:
            points.append({
                "x": x, "y": y, "t": t,
                "outsidePx": finite_number(point.get("outsidePx")),
            })

    # mtMsは全操作時間の指標としては使わず，終端サンプルの時刻としてのみ使う．
    # pointerupではtrajectoryに終端座標が追加されない場合があるため補完する．
    final_t = finite_number(row.get("mtMs"))
    final_x = finite_number(row.get("endpointX"))
    final_y = finite_number(row.get("endpointY"))
    if np.isfinite([final_t, final_x, final_y]).all() and (
        not points or final_t > max(point["t"] for point in points)
    ):
        points.append({"t": final_t, "x": final_x, "y": final_y, "outsidePx": np.nan})
    by_time = {point["t"]: point for point in points}
    return [by_time[t] for t in sorted(by_time)]


def select_analysis_trajectory(row, trajectory_col):
    """Core進入〜通過完了だけを取り出す．Core外のサンプルは補間にのみ使う．"""
    points = read_timed_trajectory(row, trajectory_col)
    if len(points) < 2 or not bool(row.get("core_completed", False)):
        return []

    try:
        entry, end = float(row["coreEntryMs"]), float(row["coreExitMs"])
    except (KeyError, TypeError, ValueError):
        return []
    if not np.isfinite([entry, end]).all() or end <= entry:
        return []

    # 同時刻のサンプルは最後の座標を使う．CSVの軌跡自体は書き換えない．
    times = np.array([point["t"] for point in points], dtype=float)
    if len(times) < 2 or entry < times[0] or end > times[-1]:
        return []
    ordered = points

    def boundary(t):
        return {
            "t": t,
            "x": float(np.interp(t, times, [point["x"] for point in ordered])),
            "y": float(np.interp(t, times, [point["y"] for point in ordered])),
        }

    return [boundary(entry)] + [point for point in ordered if entry < point["t"] < end] + [boundary(end)]


def parse_deviation_events(value):
    """有効な空リスト（逸脱なし）と，欠損・壊れたログを区別する．"""
    if isinstance(value, list):
        events = value
    else:
        try:
            events = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(events, list):
        return None
    valid = []
    for event in events:
        if not isinstance(event, dict):
            return None
        start = finite_number(event.get("startMs"))
        end = finite_number(event.get("endMs"))
        if not np.isfinite([start, end]).all() or start < 0 or end < start:
            return None
        valid.append((start, end, finite_number(event.get("maxOutsidePx"))))
    return valid


def core_trial_metrics(row, trajectory_col):
    """Coreの時間，逸脱，未完了を判定する．全区間の成功・失敗フラグは参照しない．

    逸脱時間はdeviationEventsと[coreEntryMs, coreExitMs)の共通部分．
    未完了なら試行終了までを観測区間とするが，その長さをMTには使わない．
    境界をまたぐイベントのmaxOutsidePxはCore外の最大値かもしれないので使わない．
    """
    result = {
        "core_entered": False,
        "core_completed": False,
        "core_eligible": False,
        "core_exclusion_reason": "core_not_entered",
        "core_observed_ms": np.nan,
        "core_deviation_known": False,
        "core_deviated": False,
        "core_deviation_count": np.nan,
        "core_deviation_total_ms": np.nan,
        "core_max_deviation_px": np.nan,
        "core_excessive_deviation": False,
        "core_error": False,
        "core_error_type": "unknown",
    }
    entry = finite_number(row.get("coreEntryMs"))
    end = finite_number(row.get("coreExitMs"))
    core_mt = finite_number(row.get("coreMtMs"))
    if not np.isfinite(entry) or entry < 0:
        return result
    result["core_entered"] = True
    if np.isfinite(end):
        if end <= entry or not np.isfinite(core_mt) or core_mt <= 0 or not np.isclose(
            end - entry, core_mt, atol=1.0, rtol=0.0
        ):
            result["core_exclusion_reason"] = "invalid_core_timing"
            return result
        result["core_completed"] = True
    else:
        if np.isfinite(core_mt):
            result["core_exclusion_reason"] = "missing_core_exit"
            return result
        # 未完了試行の終了時刻を区間の右端にする（全操作時間をMTに流用しない）．
        end = finite_number(row.get("mtMs"))
        if not np.isfinite(end) or end < entry:
            result["core_exclusion_reason"] = "unknown_core_observation_end"
            return result

    result["core_observed_ms"] = end - entry
    events = parse_deviation_events(row.get("deviationEvents"))
    if events is None:
        # 未完了ならERは確定する．通過完了の場合，逸脱不明をエラーなしとみなさない．
        if result["core_completed"]:
            result["core_exclusion_reason"] = "invalid_core_deviation_log"
            return result
    else:
        intervals = []
        maxima = []
        event_count = 0
        for start, stop, maximum in events:
            lo, hi = max(entry, start), min(end, stop)
            # Coreの出口時刻に始まる逸脱や，入口時刻に終了した逸脱は除く．
            overlaps = hi > lo or (entry <= start < end and start == stop)
            if not overlaps:
                continue
            event_count += 1
            if hi > lo:
                intervals.append((lo, hi))
            if entry <= start and stop <= end and np.isfinite(maximum):
                maxima.append(maximum)
        # 重複イベントでも逸脱時間を二重に数えない．
        merged = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else:
                merged.append((lo, hi))
        for point in read_timed_trajectory(row, trajectory_col):
            if entry <= point["t"] < end and np.isfinite(point["outsidePx"]):
                maxima.append(point["outsidePx"])
        # 逸脱の有無と時間はイベントログを基準にし，最大量はCore内記録のみで求める．
        maximum = max(maxima) if maxima else (0.0 if event_count == 0 else np.nan)
        deviated = event_count > 0
        if not deviated and np.isfinite(maximum) and maximum > 0:
            result["core_exclusion_reason"] = "inconsistent_core_deviation_log"
            return result
        result.update({
            "core_deviation_known": True,
            "core_deviated": deviated,
            "core_deviation_count": event_count,
            "core_deviation_total_ms": sum(hi - lo for lo, hi in merged),
            "core_max_deviation_px": maximum,
            "core_excessive_deviation": bool(np.isfinite(maximum) and maximum >= CORE_EXCESSIVE_DEVIATION_PX),
        })

    reasons = []
    if not result["core_completed"]:
        reasons.append("core_incomplete")
    if result["core_deviated"]:
        reasons.append("core_excessive_deviation" if result["core_excessive_deviation"] else "core_deviation")
    result.update({
        "core_eligible": True,
        "core_exclusion_reason": "included",
        "core_error": bool(reasons),
        "core_error_type": ";".join(reasons) if reasons else "none",
    })
    return result


def compute_trajectory_length(traj):
    points = []
    for p in traj:
        try:
            point = (float(p["x"]), float(p["y"]))
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(point).all():
            points.append(point)

    if len(points) < 2:
        return np.nan

    xy = np.asarray(points, dtype=float)
    return float(np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1)).sum())


def compute_effective_tpe_from_trajectory(df, trajectory_col, mask_col):
    """
    元コードと同じく，We=4.133*sigma_d, Ae=軌跡長, IDe=Ae/We, TPe=IDe/MTで計算する．
    水平経路の垂直方向は画面のy方向なので，sigma_dはy座標の標本標準偏差とする．
    中心線のy座標はCSVにないが，定数を引いても標準偏差は変わらない．
    軌跡の始終点を結ぶ傾斜線への射影や，x方向の端点距離は使用しない．
    MT_SOURCEと同じ時間区間の軌跡を使う．
    mask_colでTPe計算対象を指定する．
    """
    effective = pd.DataFrame(index=df.index)
    effective["sigma_d"] = np.nan
    effective["We"] = np.nan
    effective["Ae"] = np.nan
    effective["IDe"] = np.nan
    effective["TPe"] = np.nan
    effective["effective_n_points"] = np.nan

    for idx, row in df.iterrows():
        if not bool(row.get(mask_col, False)):
            continue

        mt = row.get("MT", np.nan)
        if pd.isna(mt) or mt <= 0:
            continue

        traj = select_analysis_trajectory(row, trajectory_col)
        if len(traj) < 2:
            continue

        ys = np.array([point["y"] for point in traj], dtype=float)
        sigma_d = float(np.std(ys, ddof=1))
        we = WE_COEFFICIENT * sigma_d
        ae = compute_trajectory_length(traj)

        effective.loc[idx, "sigma_d"] = sigma_d
        effective.loc[idx, "We"] = we
        effective.loc[idx, "Ae"] = ae
        effective.loc[idx, "effective_n_points"] = len(traj)
        # We=0の試行ではTPeを未定義とし，無限大のスループットを出力しない．
        if not np.isfinite(we) or we <= 0 or not np.isfinite(ae) or ae <= 0:
            continue

        ide = ae / we
        effective.loc[idx, "IDe"] = ide
        effective.loc[idx, "TPe"] = ide / mt

    return effective


def normalize_condition_from_text(x):
    """
    条件名を標準化する．
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    if s in ["c1", "1", "①", "幅広", "幅広条件", "wide", "width_wide", "w_wide"]:
        return "width_wide"
    if s in ["c3", "3", "③", "幅狭", "幅狭条件", "narrow", "width_narrow", "w_narrow"]:
        return "width_narrow"
    if s in ["c2", "2", "②", "距離長", "距離長条件", "距離大", "距離大条件", "長い", "length_long", "a_long"]:
        return "length_long"

    if "幅広" in s or "wide" in s:
        return "width_wide"
    if "幅狭" in s or "narrow" in s:
        return "width_narrow"
    if "距離長" in s or "距離大" in s or "length_long" in s or "long" in s:
        return "length_long"

    return s


def condition_from_AWID(A, W, ID):
    """
    A, W, IDの値から条件を推定する．
    """
    if pd.isna(A) or pd.isna(W):
        return np.nan

    A = float(A)
    W = float(W)

    for cond, spec in COND_SPECS.items():
        if np.isclose(A, spec["A"]) and np.isclose(W, spec["W"]):
            return cond

    return np.nan


def power_law(N, a, b, c):
    """
    Power Law of Practice
    y = a * N^(-b) + c
    """
    N = np.asarray(N, dtype=float)
    return a * np.power(N, -b) + c


def calc_np_from_b(b, p):
    """
    改善可能量のp割合に到達する試行数．
    p=0.5, 0.8, 0.9, 0.95など．
    """
    if pd.isna(b) or pd.isna(p) or b <= 0 or not (0 < p < 1):
        return np.nan

    # bが極端に小さい場合，推定到達試行数は浮動小数点で表せないほど大きくなる．
    # その場合は「有限値として推定不能」としてNaNにする．
    exponent = -np.log1p(-p) / float(b)
    if not np.isfinite(exponent) or exponent > np.log(np.finfo(float).max):
        return np.nan

    return float(np.exp(exponent))


def fit_power_law(sub, ycol, ncol="N", min_points=10):
    """
    1参加者・1条件のデータにPower Lawを当てはめる．
    """
    d = sub[[ncol, ycol]].copy()
    d[ncol] = pd.to_numeric(d[ncol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d[ncol] >= 1) & (d[ycol] > 0)]

    if len(d) < min_points or d[ncol].nunique() < min_points:
        return {
            "status": "too_few_points",
            "a": np.nan,
            "b": np.nan,
            "c": np.nan,
            "r2": np.nan,
            "aic_power": np.nan,
            "aic_constant": np.nan,
            "aic_loglinear": np.nan,
            "delta_aic_power_vs_constant": np.nan,
            "delta_aic_power_vs_loglinear": np.nan,
            "n_points": len(d),
        }

    N = d[ncol].to_numpy(dtype=float)
    y = d[ycol].to_numpy(dtype=float)

    # 初期値
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    c0 = max(np.percentile(y, 10) * 0.8, 1e-9)
    a0 = max(y_max - c0, 1e-9)
    b0 = 0.2

    # cは理論上の下限値なので，おおむね低い分位点以下に制限する
    c_upper = max(np.percentile(y, 25), 1e-9)

    try:
        popt, _ = curve_fit(
            power_law,
            N,
            y,
            p0=[a0, b0, c0],
            bounds=([0.0, 1e-6, 0.0], [np.inf, 5.0, c_upper]),
            maxfev=50000,
        )
        a, b, c = popt
        yhat = power_law(N, a, b, c)

        rss = np.sum((y - yhat) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - rss / tss if tss > 0 else np.nan

        n = len(y)
        k_power = 3
        aic_power = n * np.log(rss / n + 1e-12) + 2 * k_power

        # 定数モデル
        yhat_const = np.full_like(y, np.mean(y))
        rss_const = np.sum((y - yhat_const) ** 2)
        aic_const = n * np.log(rss_const / n + 1e-12) + 2 * 1

        # log線形モデル: y = alpha + beta log(N)
        X = np.column_stack([np.ones_like(N), np.log(N)])
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        yhat_loglin = X @ beta_hat
        rss_loglin = np.sum((y - yhat_loglin) ** 2)
        aic_loglin = n * np.log(rss_loglin / n + 1e-12) + 2 * 2

        return {
            "status": "ok",
            "a": a,
            "b": b,
            "c": c,
            "r2": r2,
            "aic_power": aic_power,
            "aic_constant": aic_const,
            "aic_loglinear": aic_loglin,
            "delta_aic_power_vs_constant": aic_power - aic_const,
            "delta_aic_power_vs_loglinear": aic_power - aic_loglin,
            "n_points": n,
        }

    except Exception as e:
        return {
            "status": f"fit_error: {e}",
            "a": np.nan,
            "b": np.nan,
            "c": np.nan,
            "r2": np.nan,
            "aic_power": np.nan,
            "aic_constant": np.nan,
            "aic_loglinear": np.nan,
            "delta_aic_power_vs_constant": np.nan,
            "delta_aic_power_vs_loglinear": np.nan,
            "n_points": len(d),
        }


def fit_power_law_signed_a(sub, ycol, ncol="N", min_points=10):
    """
    条件別の可視化用に，aの符号を制限せず y = aN^{-b}+c を当てはめる．
    MT/ERはa>0，TPeはa<0になる可能性がある．
    """
    d = sub[[ncol, ycol]].copy()
    d[ncol] = pd.to_numeric(d[ncol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d[ncol] >= 1) & (d[ycol] > 0)]

    if len(d) < min_points or d[ncol].nunique() < min_points:
        return {
            "status": "too_few_points",
            "a": np.nan,
            "b": np.nan,
            "c": np.nan,
            "r2": np.nan,
            "n_points": len(d),
        }

    d = d.sort_values(ncol)
    N = d[ncol].to_numpy(dtype=float)
    y = d[ycol].to_numpy(dtype=float)

    edge_n = max(3, int(len(d) * 0.1))
    y_start = np.nanmedian(y[:edge_n])
    y_end = np.nanmedian(y[-edge_n:])
    c0 = max(y_end, 1e-9)
    a0 = y_start - c0
    if abs(a0) < 1e-9:
        a0 = np.nanmax(y) - np.nanmin(y)
        if abs(a0) < 1e-9:
            a0 = 1e-6
    b0 = 0.2

    try:
        popt, _ = curve_fit(
            power_law,
            N,
            y,
            p0=[a0, b0, c0],
            bounds=([-np.inf, 1e-6, 0.0], [np.inf, 5.0, np.inf]),
            maxfev=50000,
        )
        a, b, c = popt
        yhat = power_law(N, a, b, c)
        rss = np.sum((y - yhat) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - rss / tss if tss > 0 else np.nan
        return {
            "status": "ok",
            "a": a,
            "b": b,
            "c": c,
            "r2": r2,
            "n_points": len(d),
        }
    except Exception as e:
        return {
            "status": f"fit_error: {e}",
            "a": np.nan,
            "b": np.nan,
            "c": np.nan,
            "r2": np.nan,
            "n_points": len(d),
        }


def summarize_by_condition(fit_df, metric):
    d = fit_df[(fit_df["metric"] == metric) & (fit_df["status"] == "ok")].copy()
    params = ["a", "b", "c", "N50", "N80", "N90", "N95"]
    out = (
        d.groupby("condition", observed=True)[params]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return out


def bootstrap_contrasts(fit_df, metric, param, boot_n=5000, seed=0):
    """
    条件間差のブートストラップ信頼区間を計算する．
    """
    rng = np.random.default_rng(seed)
    d = fit_df[(fit_df["metric"] == metric) & (fit_df["status"] == "ok")].copy()

    values = {}
    for cond in COND_ORDER:
        arr = d.loc[d["condition"] == cond, param].dropna().to_numpy()
        values[cond] = arr[np.isfinite(arr)]

    rows = []
    active_contrasts = [
        (name, {left: 1.0, right: -1.0}, False)
        for name, _, left, right in CONTRAST_DEFS
    ]
    active_contrasts.extend([
        ("abs_width_effect_minus_abs_length_effect",
         {"width_wide": 1.0, "width_narrow": -1.0, "length_long": 1.0}, True),
        ("ID18_mean_minus_ID27_mean",
         {"width_wide": 1.0, "length_long": -0.5, "width_narrow": -0.5}, False),
    ])

    for name, weights, absolute_effect in active_contrasts:
        if any(len(values[cond]) == 0 for cond in weights):
            continue

        def contrast_value(means):
            if absolute_effect:
                width_diff = means["width_wide"] - means["width_narrow"]
                length_diff = means["length_long"] - means["width_wide"]
                return abs(width_diff) - abs(length_diff)
            return sum(weights[cond] * means[cond] for cond in weights)

        row = {
            "metric": metric,
            "param": param,
            "contrast": name,
            "mean": contrast_value({cond: values[cond].mean() for cond in weights}),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_two_sided_approx": np.nan,
            "status": "too_few_participants",
        }
        # 1参加者では参加者間の不確実性を推定できないため，差のみ出力する．
        if all(len(values[cond]) >= 2 for cond in weights):
            samples = {
                cond: rng.choice(values[cond], size=(boot_n, len(values[cond])), replace=True).mean(axis=1)
                for cond in weights
            }
            arr = contrast_value(samples)
            row.update({
                "mean": arr.mean(),
                "ci_low": np.percentile(arr, 2.5),
                "ci_high": np.percentile(arr, 97.5),
                "p_two_sided_approx": min(1.0, 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))),
                "status": "ok",
            })
        rows.append(row)

    return pd.DataFrame(rows)


# 各フェーズを分け，試行単位の平均・不偏分散と実際の区間範囲を集計する．
def build_pilot_summary(data, bin_size=None):
    values = data.copy()
    values["MT_value"] = values["MT"].where(values["success_for_mt"])
    values["TPe_value"] = values["TPe_clean"].where(values["success_clean_for_tpe"])
    values["ER_value"] = values["error_counted"].astype(float)
    keys = ["phase", "condition"]
    if bin_size is not None:
        values["bin"] = ((values["N"] - 1) // bin_size) + 1
        keys.append("bin")
    summary = values.groupby(keys, observed=True, sort=True).agg(
        n_participants=("participant", "nunique"),
        n_trials=("N", "size"),
        N_first=("N", "min"),
        N_last=("N", "max"),
        N_center=("N", "mean"),
        n_errors=("error_counted", "sum"),
        MT_n=("MT_value", "count"),
        MT_mean_s=("MT_value", "mean"),
        MT_variance_s2=("MT_value", "var"),
        ER_mean=("ER_value", "mean"),
        ER_variance_01=("ER_value", "var"),
        TPe_n=("TPe_value", "count"),
        TPe_mean_per_s=("TPe_value", "mean"),
        TPe_variance_per_s2=("TPe_value", "var"),
    ).reset_index()
    summary["ER_mean_percent"] = summary.pop("ER_mean") * 100
    summary["condition_label"] = summary["condition"].astype(str).map(make_condition_metric_label)
    if bin_size is not None:
        summary["bin_size"] = bin_size
        summary["bin_start"] = (summary["bin"] - 1) * bin_size + 1
        summary["bin_end"] = summary["bin"] * bin_size
        summary["bin_complete"] = summary["n_trials"].eq(summary["n_participants"] * bin_size)
    return summary.sort_values(
        "phase", key=lambda values: values.map({"practice": 0, "pilot": 1, "post": 2}).fillna(3),
        kind="stable",
    ).reset_index(drop=True)


def plot_pilot_summary(summary, phase, bin_size=None):
    data = summary[summary["phase"].eq(phase)]
    conditions = [cond for cond in COND_ORDER if data["condition"].eq(cond).any()]
    colors = dict(zip(COND_ORDER, ["tab:blue", "tab:green", "tab:orange"]))
    metrics = [
        ("MT_mean_s", "MT_variance_s2", "MT", "Mean [s]", "Variance [s²]", "MT_n"),
        ("ER_mean_percent", "ER_variance_01", "ER", "Mean [%]", "Variance of 0/1 errors", "n_trials"),
        ("TPe_mean_per_s", "TPe_variance_per_s2", "TPe", "Mean [1/s]", "Variance [1/s²]", "TPe_n"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for col, (mean_col, var_col, metric, mean_label, var_label, count_col) in enumerate(metrics):
        for row, (value_col, ylabel) in enumerate([(mean_col, mean_label), (var_col, var_label)]):
            ax = axes[row, col]
            for position, cond in enumerate(conditions):
                sub = data[data["condition"].eq(cond)].sort_values("N_center")
                label = make_condition_metric_label(cond)
                if bin_size is None:
                    value = sub[value_col].iloc[0]
                    if pd.notna(value):
                        ax.bar(position, value, color=colors[cond], width=0.6)
                    count = int(sub[count_col].iloc[0])
                    ax.text(position, 0.98, f"n={count}" if pd.notna(value) else f"n={count}, NA",
                            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
                else:
                    ax.plot(sub["N_center"], sub[value_col], color=colors[cond], marker="o",
                            linewidth=1.5, markersize=5, label=label)
                    partial = sub[~sub["bin_complete"]]
                    ax.scatter(partial["N_center"], partial[value_col], facecolors="white",
                               edgecolors=colors[cond], s=38, zorder=3)
            if bin_size is None:
                ax.set_xticks(range(len(conditions)), [f"C{COND_NUMBER[cond]}" for cond in conditions])
                ax.set_xlabel("Condition")
            else:
                ax.set_xlim(0, data["N_last"].max() + 1)
                ax.set_xlabel("Core attempt N within phase and condition")
            ax.set_title(f"{metric}: {'mean' if row == 0 else 'sample variance'}")
            ax.set_ylabel(ylabel)
            upper = data[value_col].max()
            ax.set_ylim(0, upper * 1.2 if pd.notna(upper) and upper > 0 else 1)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
    title = {"practice": "Practice", "pilot": "Pilot", "post": "Post"}.get(phase, phase)
    scope = "condition summaries" if bin_size is None else f"{bin_size}-attempt bins"
    fig.suptitle(f"{title}: Core MT, ER and TPe — {scope}", fontsize=16, y=0.98)
    if bin_size is None:
        caption = "  |  ".join(make_condition_metric_label(cond) for cond in conditions)
        fig.text(0.5, 0.925, caption, ha="center", fontsize=9)
    else:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
                   ncol=len(conditions), fontsize=9, frameon=False)
    note = "Variance: ddof=1; fewer than 2 observations = NA. ER variance uses 0/1 errors."
    if bin_size is not None:
        note += " Open circles: partial bins."
        if data["bin"].max() == 1:
            note += " One bin per condition."
    fig.text(0.5, 0.02, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.91))
    phase_name = "".join(char if char.isalnum() or char in "_-" else "_" for char in str(phase))
    suffix = "condition_summary" if bin_size is None else f"bin{bin_size}_mean_variance"
    save_png_figure(fig, OUT_DIR / f"{phase_name}_{suffix}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_mt_bins(bins):
    data = bins[bins["bin_size"].eq(10) & bins["condition"].isin(["length_long", "width_narrow"])]
    colors = {"length_long": "tab:green", "width_narrow": "tab:orange"}
    for phase, phase_data in data.groupby("phase", sort=False, observed=True):
        phase_label = {"practice": "反復試行", "pilot": "予備試行", "post": "事後試行"}.get(phase, phase)
        max_n = phase_data["N_last"].max()
        max_mt = phase_data["MT_mean_s"].max()
        for cond, color in colors.items():
            sub = phase_data[phase_data["condition"].eq(cond)].sort_values("N_center")
            if sub.empty:
                continue
            spec = COND_SPECS[cond]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(sub["N_center"], sub["MT_mean_s"], color=color,
                    marker="o", markersize=5, linewidth=1.8)
            partial = sub[~sub["bin_complete"]]
            ax.scatter(partial["N_center"], partial["MT_mean_s"], facecolors="white",
                       edgecolors=color, s=42, zorder=3)
            ax.set_title(f"C{spec['number']}：{phase_label}（10試行平均）", fontsize=15, pad=14)
            ax.set_xlabel("試行回数", fontsize=12)
            ax.set_ylabel("平均MT［s］", fontsize=12)
            xmax = int(np.ceil(max_n / 10) * 10)
            ax.set_xlim(0, xmax)
            ax.set_xticks(np.arange(0, xmax + 1, 20 if xmax > 100 else 5))
            ax.set_ylim(0, max_mt * 1.15 if pd.notna(max_mt) and max_mt > 0 else 1)
            ax.grid(alpha=0.25)
            ax.set_axisbelow(True)
            fig.text(0.5, 0.025, "各点：10試行区間の中央・除外基準適用後の平均MT ／ 白抜き：末尾の端数区間",
                     ha="center", fontsize=9)
            fig.tight_layout(rect=(0, 0.07, 1, 1))
            phase_name = "".join(char if char.isalnum() or char in "_-" else "_" for char in str(phase))
            save_png_figure(fig, OUT_DIR / f"{phase_name}_C{spec['number']}_MT_bin10.png",
                            dpi=200, bbox_inches="tight", pad_inches=0.15)
            plt.close(fig)


def write_pilot_summary(data):
    summary = build_pilot_summary(data)
    bins = pd.concat([build_pilot_summary(data, size) for size in BIN_SIZES], ignore_index=True)
    summary.to_csv(OUT_DIR / "condition_summary.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(OUT_DIR / "bin_summary.csv", index=False, encoding="utf-8-sig")
    plot_mt_bins(bins)
    for phase in summary["phase"].unique():
        plot_pilot_summary(summary, phase)
        for bin_size in BIN_SIZES:
            plot_pilot_summary(bins[bins["bin_size"].eq(bin_size)], phase, bin_size)
    print("\nCore metric means and sample variances by phase and condition:")
    print(summary.to_string(index=False))
    print(f"\nOutput complete: {OUT_DIR.resolve()}")


# ============================================================
# 2. CSV読み込み
# ============================================================

setup_matplotlib()

args = parse_args()
apply_cli_overrides(args)
MT_SOURCE = args.mt_source
ANALYSIS_PHASE = args.phase
if args.pilot_summary:
    OUT_DIR = OUT_DIR / "pilot_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)
if not args.pilot_summary:
    CONDITION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
csv_files = find_csv_files(args.csv_filenames, read_all=args.all)
if not csv_files:
    raise FileNotFoundError(f"CSV directory not found: {CSV_DIR.resolve()}")

dfs = []
for fp in csv_files:
    tmp = pd.read_csv(fp, dtype={"participantId": "string"})
    tmp["source_file"] = fp.name
    dfs.append(tmp)

raw = pd.concat(dfs, ignore_index=True)
print(f"Loaded: {len(raw)} rows, {len(csv_files)} files")
print("CSV files:")
for fp in csv_files:
    print(f"  - {fp.name}")
print("Analysis settings:")
print(f"  - MT_SOURCE = {MT_SOURCE}")
print(f"  - ANALYSIS_SCOPE = {ANALYSIS_SCOPE}")
print(f"  - ER denominator = {CORE_TRIAL_DENOMINATOR}")
print(f"  - ANALYSIS_PHASE = {ANALYSIS_PHASE}")
print(f"  - DEVIATION_EXCLUSION_RATIO = {DEVIATION_EXCLUSION_RATIO:g}")
print(f"  - INCLUDE_DEVIATED_TRIALS = {INCLUDE_DEVIATED_TRIALS}")
print(f"  - DELETE_OUTLIER = {DELETE_OUTLIER}")
print(f"  - OUTLIER_SIGMA = {OUTLIER_SIGMA:g}")


# ============================================================
# 3. 列名の標準化
# ============================================================

df = raw.copy()
if args.pilot_summary:
    if "phase" not in df.columns:
        raise KeyError("--pilot-summary requires a phase column to separate practice and post trials.")
    df["phase"] = df["phase"].astype("string").str.strip().str.lower()
    if df["phase"].isna().any() or df["phase"].eq("").any():
        raise ValueError("--pilot-summary requires a nonempty phase for every trial.")

# 反復試行と事後試行を区別してから，従来と同じ分析を行う．
if "phase" in df.columns and ANALYSIS_PHASE != "all":
    selected_phases = ["practice", "pilot"] if ANALYSIS_PHASE == "learning" else [ANALYSIS_PHASE]
    phase_mask = df["phase"].astype(str).str.strip().str.lower().isin(selected_phases)
    df = df.loc[phase_mask].copy()
elif "phase" not in df.columns and ANALYSIS_PHASE not in ["learning", "all"]:
    raise ValueError("The selected phase requires a phase column in the CSV.")
if df.empty:
    raise ValueError(f"No rows match phase={ANALYSIS_PHASE}.")

participant_col = pick_col(df, [
    "participantId", "participant_id", "participant", "user_id", "subject", "subject_id",
    "worker_id", "pid", "参加者ID", "参加者"
], required=False)

condition_col = pick_col(df, [
    "conditionId", "condition", "condition_id", "cond", "group", "条件"
], required=False)

trial_col = pick_col(df, [
    "trialInCondition", "trialInPhase", "totalTrial",
    "trial", "trial_index", "trial_num", "trial_number", "N", "試行", "試行番号"
], required=False)

required_core = ["coreMtMs", "coreEntryMs", "coreExitMs", "deviationEvents"]
missing_core = [col for col in required_core if col not in df.columns]
if missing_core:
    raise KeyError(f"Core-only analysis requires columns: {missing_core}")

a_col = pick_col(df, ["amplitude", "A", "a", "distance", "path_length", "length", "経路長", "距離"], required=False)
w_col = pick_col(df, ["W", "w", "width", "path_width", "経路幅", "幅"], required=False)
id_col = pick_col(df, ["steeringId", "ID", "id", "difficulty", "difficulty_index", "D", "難易度"], required=False)

trajectory_col = pick_col(df, ["trajectoryJson", "trajectory", "trajectory_json", "軌跡"], required=True)


if participant_col is None:
    # 参加者IDがない場合，ファイル単位で参加者とみなす
    df["participant"] = df["source_file"]
else:
    df["participant"] = df[participant_col].astype(str)

if trial_col is None:
    df["trial_original"] = np.arange(len(df))
else:
    df["trial_original"] = pd.to_numeric(df[trial_col], errors="coerce")

# すべてのMT集計・外れ値判定・回帰がこのCore MT列を使う．
df["MT"] = pd.to_numeric(df["coreMtMs"], errors="coerce") / 1000.0

if a_col is not None:
    df["A"] = pd.to_numeric(df[a_col], errors="coerce")
else:
    df["A"] = np.nan

if w_col is not None:
    df["W"] = pd.to_numeric(df[w_col], errors="coerce")
else:
    df["W"] = np.nan

if id_col is not None:
    df["ID"] = pd.to_numeric(df[id_col], errors="coerce")
else:
    df["ID"] = df["A"] / df["W"]

if condition_col is not None:
    df["condition"] = df[condition_col].apply(normalize_condition_from_text)
else:
    df["condition"] = [
        condition_from_AWID(A, W, ID)
        for A, W, ID in zip(df["A"], df["W"], df["ID"])
    ]

# A, Wから条件が明確に推定できる場合は補正
mask_unknown_cond = df["condition"].isna() | ~df["condition"].isin(COND_ORDER)
df.loc[mask_unknown_cond, "condition"] = [
    condition_from_AWID(A, W, ID)
    for A, W, ID in zip(
        df.loc[mask_unknown_cond, "A"],
        df.loc[mask_unknown_cond, "W"],
        df.loc[mask_unknown_cond, "ID"]
    )
]

# C番号が旧実験などの異なるA/Wを示している場合は，誤分類を防ぐ．
for cond, spec in COND_SPECS.items():
    cond_mask = df["condition"].eq(cond)
    for col in ["A", "W", "ID"]:
        mismatch = cond_mask & df[col].notna() & ~np.isclose(df[col], spec[col])
        if mismatch.any():
            raise ValueError(f"C{spec['number']} has unexpected {col}; expected {spec[col]} for horizontal steering.")
        df.loc[cond_mask & df[col].isna(), col] = spec[col]


# ============================================================
# 4. Core区間でのエラー処理・分析対象判定
# ============================================================

core_metrics = pd.DataFrame(
    [core_trial_metrics(row, trajectory_col) for _, row in df.iterrows()],
    index=df.index,
)
for col in core_metrics.columns:
    df[col] = core_metrics[col]

# 元CSVのsuccess/errorType/deviatedは参照用にそのまま保持する．
# Core通過後の離脱は失敗にしない．未完了はERの分母・分子に残し，MT/TPeは欠損にする．
df["is_counted_trial"] = df["core_eligible"]
df["error_counted"] = df["core_error"]
df["error_text"] = df["core_error_type"]
df["MT"] = df["MT"].where(df["core_completed"] & df["core_eligible"])
audit_columns = [
    "source_file", "participant", "condition", "phase", "trial_original", "totalTrial",
    "attemptNumber", "countedAsTrial", "success", "errorType", "coreEntryMs", "coreExitMs",
    "coreMtMs", *core_metrics.columns,
]
core_audit = df[[col for col in audit_columns if col in df.columns]].copy()
core_audit.to_csv(OUT_DIR / "core_trial_audit.csv", index=False, encoding="utf-8-sig")
core_exclusion_counts = df.loc[~df["core_eligible"], "core_exclusion_reason"].value_counts()
invalid_entered = df["core_entered"] & ~df["core_eligible"]
if invalid_entered.any():
    # Core進入試行を黙ってERの分母から落とすことを防ぐ．入力CSVの補完・改変はしない．
    reasons = df.loc[invalid_entered, "core_exclusion_reason"].value_counts().to_dict()
    raise ValueError(f"Core-entered trials contain invalid logs: {reasons}. See output/core_trial_audit.csv.")
df = df[df["is_counted_trial"]].copy()

# 水平版の既知条件だけを対象にする．未対応の条件を黙って欠損カテゴリにしない．
unknown_condition = ~df["condition"].isin(COND_ORDER)
if unknown_condition.any():
    raise ValueError("Unsupported horizontal condition. Expected C1 (540/30), C2 (810/30), or C3 (540/20).")
if df.empty:
    raise ValueError("No core-entered trials remain for the selected CSV files and phase.")

# 条件順を固定
df["condition"] = pd.Categorical(df["condition"], categories=COND_ORDER, ordered=True)
df["condition"] = df["condition"].cat.remove_unused_categories()

# Core進入試行を実施順で並べる．Core内未完了のやり直しも1試行として数える．
# 事後試行ではtrialInConditionが再び1から始まるため，全体の実施順を使う．
for col in ["totalTrial", "trialInPhase", "attemptNumber"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df["started_at"] = (
    pd.to_datetime(df["startedAtIso"], errors="coerce", utc=True)
    if "startedAtIso" in df.columns else pd.NaT
)
sort_cols = ["participant", "condition"]
if df["started_at"].notna().all():
    sort_cols.append("started_at")
sort_cols.append("source_file")
sort_cols.extend(col for col in ["totalTrial", "trialInPhase", "trial_original", "attemptNumber"] if col in df.columns)
df = df.sort_values(sort_cols, kind="stable").copy()
trial_groups = ["participant", "condition"]
if args.pilot_summary:
    trial_groups.append("phase")
df["N"] = df.groupby(trial_groups, observed=True).cumcount() + 1

# 400試行より後がある場合は除外
df = df[df["N"] <= MAX_TRIAL_N].copy()

# 完了試行はcoreMtMs，未完了はCore進入〜終了の観測時間を分母にする．
# 未完了試行はこの割合によらずMT/TPe対象外だが，ERには必ず残す．
core_duration_ms = pd.to_numeric(df["coreMtMs"], errors="coerce").where(
    df["core_completed"], df["core_observed_ms"]
)
core_duration_ms = core_duration_ms.where(core_duration_ms > 0)
df["deviation_time_ratio"] = df["core_deviation_total_ms"] / core_duration_ms
df["exclude_by_deviation_ratio"] = (
    df["deviation_time_ratio"].notna()
    & (df["deviation_time_ratio"] >= DEVIATION_EXCLUSION_RATIO)
)

df["performance_trial_before_outlier"] = build_performance_trial_mask(df)
if args.pilot_summary:
    df["is_mt_outlier"] = False
    for _, phase_data in df.groupby("phase", observed=True):
        df.loc[phase_data.index, "is_mt_outlier"] = mark_mt_outliers(
            phase_data, phase_data["performance_trial_before_outlier"],
        )
else:
    df["is_mt_outlier"] = mark_mt_outliers(df, df["performance_trial_before_outlier"])
df["success_for_mt"] = df["performance_trial_before_outlier"] & ~df["is_mt_outlier"]

# 主分析: MT分析と同じ基準を使う．
# Core内の逸脱設定・逸脱時間割合・Core MT外れ値判定を適用する．
df["success_clean_for_tpe"] = df["success_for_mt"]

# 補助分析: Core通過完了ならCore内の逸脱を含める（逸脱時間割合・外れ値判定も適用）．
df["success_with_deviation_for_tpe"] = (
    df["core_completed"]
    & df["MT"].notna()
    & (df["MT"] > 0)
    & ~df["exclude_by_deviation_ratio"]
    & ~df["is_mt_outlier"]
)

# ブロック
df["block20"] = ((df["N"] - 1) // BIN_SIZE) + 1
df["rest_block100"] = ((df["N"] - 1) // 100) + 1
df["logN"] = np.log(df["N"])

# TPe = (Core軌跡のAe / Core軌跡のWe) / Core MT．既存の全区間Ae/We/TPeは使わない．
effective_clean = compute_effective_tpe_from_trajectory(df, trajectory_col, "success_clean_for_tpe")
effective_with_deviation = compute_effective_tpe_from_trajectory(df, trajectory_col, "success_with_deviation_for_tpe")
for col in ["sigma_d", "We", "Ae", "IDe", "TPe", "effective_n_points"]:
    df[col] = effective_clean[col]
df["TPe_clean"] = effective_clean["TPe"]
df["TPe_with_deviation"] = effective_with_deviation["TPe"]

df["inv_TPe"] = 1 / df["TPe"]
df["inv_TPe_with_deviation"] = 1 / df["TPe_with_deviation"]

df.to_csv(OUT_DIR / "preprocessed_trials.csv", index=False, encoding="utf-8-sig")
trial_handling_summary = pd.DataFrame([{
    "MT_SOURCE": MT_SOURCE,
    "ANALYSIS_SCOPE": ANALYSIS_SCOPE,
    "ER_DENOMINATOR": CORE_TRIAL_DENOMINATOR,
    "ER_DEFINITION": "core_incomplete_or_core_deviation",
    "CORE_EXCESSIVE_DEVIATION_PX": CORE_EXCESSIVE_DEVIATION_PX,
    "ANALYSIS_PHASE": ANALYSIS_PHASE,
    "MT_REQUIRES_CORE_COMPLETION": True,
    "INCLUDE_DEVIATED_TRIALS": INCLUDE_DEVIATED_TRIALS,
    "DEVIATION_EXCLUSION_RATIO": DEVIATION_EXCLUSION_RATIO,
    "DELETE_OUTLIER": DELETE_OUTLIER,
    "OUTLIER_SIGMA": OUTLIER_SIGMA,
    "n_counted_trials": len(df),
    "n_core_not_entered": int(core_exclusion_counts.get("core_not_entered", 0)),
    "n_core_completed": int(df["core_completed"].sum()),
    "n_core_incomplete": int((~df["core_completed"]).sum()),
    "n_core_deviated": int(df["core_deviated"].sum()),
    "n_core_deviation_unknown": int((~df["core_deviation_known"]).sum()),
    "n_error_counted": int(df["error_counted"].sum()),
    "n_excluded_by_deviation_ratio": int(df["exclude_by_deviation_ratio"].sum()),
    "n_performance_before_outlier": int(df["performance_trial_before_outlier"].sum()),
    "n_mt_outlier": int(df["is_mt_outlier"].sum()),
    "n_performance_after_outlier": int(df["success_for_mt"].sum()),
    "n_success_clean_for_tpe": int(df["success_clean_for_tpe"].sum()),
    "n_success_with_deviation_for_tpe": int(df["success_with_deviation_for_tpe"].sum()),
    "n_TPe_clean": int(df["TPe_clean"].notna().sum()),
    "n_TPe_with_deviation": int(df["TPe_with_deviation"].notna().sum()),
}])
if args.pilot_summary:
    trial_handling_summary["TRIAL_INDEX_SCOPE"] = "participant_condition_phase"
    trial_handling_summary["MT_OUTLIER_SCOPE"] = "phase_all_conditions"
    trial_handling_summary["VARIANCE_DDOF"] = 1
    trial_handling_summary["ER_VARIANCE_SCALE"] = "binary_0_1"
    trial_handling_summary["SUMMARY_BIN_SIZES"] = ",".join(map(str, BIN_SIZES))
trial_handling_summary.to_csv(
    OUT_DIR / "trial_handling_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
print(f"Preprocessed data: {len(df)} rows")
print(df.groupby("condition", observed=True)["participant"].nunique())
print(
    "MT/TPe analysis rows:",
    int(df["success_for_mt"].sum()),
    "rows",
    f"(MT outliers removed: {int(df['is_mt_outlier'].sum())} rows)",
)
print(
    "TPe main-analysis rows:",
    int(df["TPe_clean"].notna().sum()),
    "rows",
)
print(
    "TPe auxiliary-analysis rows:",
    int(df["TPe_with_deviation"].notna().sum()),
    "rows",
)


if args.pilot_summary:
    write_pilot_summary(df)
    raise SystemExit(0)


# ============================================================
# 5. 記述統計
# ============================================================

desc_trial = (
    df.groupby(["condition", "rest_block100"], observed=True)
    .agg(
        n_trials=("N", "count"),
        n_participants=("participant", "nunique"),
        MT_success_mean=("MT", lambda x: np.nan),
        ER=("error_counted", "mean"),
    )
    .reset_index()
)

# 成功試行MTを別途集計
mt_success_desc = (
    df[df["success_for_mt"]]
    .groupby(["condition", "rest_block100"], observed=True)
    .agg(
        MT_mean=("MT", "mean"),
        MT_median=("MT", "median"),
        MT_sd=("MT", "std"),
    )
    .reset_index()
)

desc = desc_trial.merge(mt_success_desc, on=["condition", "rest_block100"], how="left")
desc.to_csv(OUT_DIR / "descriptive_by_100_trials.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 6. Power Lawフィット
# ============================================================

fit_rows = []

# 6-1. MT: 成功試行のみ
mt_df = df[df["success_for_mt"]].copy()

for (pid, cond), sub in mt_df.groupby(["participant", "condition"], observed=True):
    res = fit_power_law(sub, ycol="MT", ncol="N")
    row = {
        "participant": pid,
        "condition": cond,
        "metric": "MT_success",
        **res
    }
    fit_rows.append(row)

# 6-2. 1/TPe: TPeがある場合のみ
if df["inv_TPe"].notna().sum() > 0:
    inv_tpe_df = df[df["success_clean_for_tpe"] & df["inv_TPe"].notna()].copy()
    for (pid, cond), sub in inv_tpe_df.groupby(["participant", "condition"], observed=True):
        res = fit_power_law(sub, ycol="inv_TPe", ncol="N")
        row = {
            "participant": pid,
            "condition": cond,
            "metric": "inv_TPe",
            **res
        }
        fit_rows.append(row)

# 6-2補助. 逸脱あり成功試行も含めた1/TPe
if df["inv_TPe_with_deviation"].notna().sum() > 0:
    inv_tpe_dev_df = df[
        df["success_with_deviation_for_tpe"]
        & df["inv_TPe_with_deviation"].notna()
    ].copy()
    for (pid, cond), sub in inv_tpe_dev_df.groupby(["participant", "condition"], observed=True):
        res = fit_power_law(sub, ycol="inv_TPe_with_deviation", ncol="N")
        row = {
            "participant": pid,
            "condition": cond,
            "metric": "inv_TPe_with_deviation",
            **res
        }
        fit_rows.append(row)

# 6-3. ER: 20試行ごとのエラー率にフィット
er20 = (
    df.groupby(["participant", "condition", "block20"], observed=True)
    .agg(
        N_mid=("N", "mean"),
        error_rate=("error_counted", "mean"),
        error_count=("error_counted", "sum"),
        n=("error_counted", "size"),
    )
    .reset_index()
)

# 0のERはPower Lawの制約上そのままだと扱いにくいので，微小値を足す
er20["error_rate_for_fit"] = er20["error_rate"].clip(lower=1e-4)

for (pid, cond), sub in er20.groupby(["participant", "condition"], observed=True):
    res = fit_power_law(sub, ycol="error_rate_for_fit", ncol="N_mid", min_points=6)
    row = {
        "participant": pid,
        "condition": cond,
        "metric": "ER_block20",
        **res
    }
    fit_rows.append(row)

fit_df = pd.DataFrame(fit_rows)

# N50/N80/N90/N95．within_400は400試行以内の到達予測であり，観測済みかどうかではない．
for p, name in [(0.50, "N50"), (0.80, "N80"), (0.90, "N90"), (0.95, "N95")]:
    fit_df[name] = fit_df["b"].apply(lambda b: calc_np_from_b(b, p))

fit_df["N50_within_400"] = fit_df["N50"] <= 400
fit_df["N80_within_400"] = fit_df["N80"] <= 400
fit_df["N90_within_400"] = fit_df["N90"] <= 400
fit_df["N95_within_400"] = fit_df["N95"] <= 400

fit_df.to_csv(OUT_DIR / "power_law_fit_by_participant.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 7. 条件別集計
# ============================================================

summary_rows = []

for metric in fit_df["metric"].dropna().unique():
    d = fit_df[(fit_df["metric"] == metric) & (fit_df["status"] == "ok")].copy()
    for cond, sub in d.groupby("condition", observed=True):
        for param in ["a", "b", "c", "N50", "N80", "N90", "N95", "r2",
                      "delta_aic_power_vs_constant", "delta_aic_power_vs_loglinear"]:
            vals = sub[param].dropna()
            summary_rows.append({
                "metric": metric,
                "condition": cond,
                "condition_label": COND_DISPLAY.get(str(cond), str(cond)),
                "param": param,
                "mean": vals.mean(),
                "sd": vals.std(),
                "median": vals.median(),
                "n": len(vals),
            })

summary_df = pd.DataFrame(summary_rows, columns=[
    "metric", "condition", "condition_label", "param", "mean", "sd", "median", "n",
])
summary_df.to_csv(OUT_DIR / "summary_fit_params_by_condition.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 8. ブートストラップによる条件間比較
# ============================================================

contrast_all = []

for metric in fit_df["metric"].dropna().unique():
    for param in ["a", "b", "c", "N50", "N80", "N90", "N95"]:
        cdf = bootstrap_contrasts(fit_df, metric=metric, param=param, boot_n=BOOT_N, seed=42)
        if len(cdf) > 0:
            contrast_all.append(cdf)

if contrast_all:
    contrast_df = pd.concat(contrast_all, ignore_index=True)
    contrast_df.to_csv(OUT_DIR / "bootstrap_contrasts.csv", index=False, encoding="utf-8-sig")
else:
    contrast_df = pd.DataFrame()


# ============================================================
# 9. 混合効果モデル・GEE
# ============================================================

model_texts = [
    "Analysis scope: Core interval only. MT = coreMtMs / 1000; "
    "Core ER = (Core-incomplete or Core-deviated attempts) / Core-entered attempts. "
    "N counts Core-entered attempts, including unfinished retries."
]

# 9-1. MT: 成功試行のみ，log(MT) ~ logN * condition
mt_model_df = df[df["success_for_mt"]].copy()
mt_model_df = mt_model_df[mt_model_df["MT"] > 0].copy()
mt_model_df["logMT"] = np.log(mt_model_df["MT"])

def append_mixed_model_summary(data, label, random_slope=False):
    model_texts.append(f"===== {label} =====")
    if data["participant"].nunique() < 2:
        model_texts.append("Skipped: at least two participants are required for a participant-level mixed model.")
        return
    model_data = data.copy()
    model_data["condition"] = model_data["condition"].cat.remove_unused_categories()
    formulas = ["~logN", "1"] if random_slope else ["1"]
    for re_formula in formulas:
        try:
            md = smf.mixedlm(
                "logMT ~ logN * C(condition)",
                data=model_data,
                groups=model_data["participant"],
                re_formula=re_formula,
            )
            mdf = md.fit(method="lbfgs", maxiter=2000)
            model_texts.append(f"Random effects: {re_formula}")
            model_texts.append(str(mdf.summary()))
            return
        except Exception as error:
            model_texts.append(f"MixedLM failed ({re_formula}): {error}")


append_mixed_model_summary(mt_model_df, "MixedLM: logMT ~ logN * condition", random_slope=True)

# 9-2. ER: GEE binomial
gee_df = df.copy()
gee_df["error_int"] = gee_df["error_counted"].astype(int)

if gee_df["participant"].nunique() < 2:
    model_texts.append("GEE skipped: at least two participant clusters are required.")
elif gee_df["error_int"].nunique() < 2:
    model_texts.append("GEE skipped: both error and non-error trials are required.")
else:
    try:
        gee = smf.gee(
            "error_int ~ logN * C(condition)",
            groups="participant",
            data=gee_df,
            family=sm.families.Binomial(),
        ).fit()
        model_texts.append("\n===== GEE Binomial: error ~ logN * condition =====")
        model_texts.append(str(gee.summary()))
    except Exception as e:
        model_texts.append("GEE failed:")
        model_texts.append(str(e))

# 9-3. 同一ID比較（水平版ではC2とC3がID27）
for label, conds in SAME_ID_PAIRS:
    sub = mt_model_df[mt_model_df["condition"].isin(conds)].copy()
    sub["condition"] = sub["condition"].cat.remove_unused_categories()
    if len(sub) > 0 and sub["condition"].nunique() == 2:
        append_mixed_model_summary(sub, f"Same ID comparison: {label}")
    else:
        model_texts.append(f"Same ID model skipped ({label}): both C2 and C3 need valid MT trials.")

with open(OUT_DIR / "model_summaries.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(model_texts))


# ============================================================
# 10. 図の出力
# ============================================================

def sem(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return np.nan
    return x.std() / np.sqrt(len(x))


def plot_block_metric(data, ycol, ylabel, filename, success_only=False):
    d = data.copy()

    agg = (
        d.groupby(["condition", "block20"], observed=True)
        .agg(
            N_mid=("N", "mean"),
            mean=(ycol, "mean"),
            se=(ycol, sem),
        )
        .reset_index()
    )

    plt.figure(figsize=(9, 5))

    for cond in COND_ORDER:
        sub = agg[agg["condition"] == cond]
        if len(sub) == 0:
            continue
        plt.errorbar(
            sub["N_mid"],
            sub["mean"],
            yerr=sub["se"],
            marker="o",
            linewidth=1,
            capsize=2,
            label=COND_DISPLAY.get(cond, cond),
        )

    plt.xlabel("Core attempt number N")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_png_figure(plt.gcf(), OUT_DIR / filename, dpi=200)
    plt.close()


def build_metric_bin_summary(data, bin_sizes):
    frames = []

    for bin_size in bin_sizes:
        d = data.copy()
        d["binSize"] = bin_size
        d["bin"] = ((d["N"] - 1) // bin_size) + 1
        d["binStart"] = (d["bin"] - 1) * bin_size + 1
        d["binEnd"] = d["bin"] * bin_size
        d["binCenter"] = (d["binStart"] + d["binEnd"]) / 2

        keys = ["condition", "binSize", "bin", "binStart", "binEnd", "binCenter"]

        err = (
            d.groupby(keys, observed=True)
            .agg(
                n_trials=("N", "count"),
                n_error=("error_counted", "sum"),
                ER_mean=("error_counted", "mean"),
                ER_se=("error_counted", sem),
            )
            .reset_index()
        )
        err["ER_mean"] = err["ER_mean"] * 100
        err["ER_se"] = err["ER_se"] * 100

        mt = (
            d[d["success_for_mt"]]
            .groupby(keys, observed=True)
            .agg(
                MT_n=("MT", "count"),
                MT_mean=("MT", "mean"),
                MT_se=("MT", sem),
            )
            .reset_index()
        )

        tpe = (
            d[d["success_clean_for_tpe"] & d["TPe_clean"].notna()]
            .groupby(keys, observed=True)
            .agg(
                TPe_n=("TPe_clean", "count"),
                TPe_mean=("TPe_clean", "mean"),
                TPe_se=("TPe_clean", sem),
            )
            .reset_index()
        )

        tpe_dev = (
            d[d["success_with_deviation_for_tpe"] & d["TPe_with_deviation"].notna()]
            .groupby(keys, observed=True)
            .agg(
                TPe_with_deviation_n=("TPe_with_deviation", "count"),
                TPe_with_deviation_mean=("TPe_with_deviation", "mean"),
                TPe_with_deviation_se=("TPe_with_deviation", sem),
            )
            .reset_index()
        )

        out = err.merge(mt, on=keys, how="outer")
        out = out.merge(tpe, on=keys, how="outer")
        out = out.merge(tpe_dev, on=keys, how="outer")
        out["condition_label"] = out["condition"].astype(str).map(make_condition_metric_label)
        frames.append(out.sort_values(["condition", "binSize", "bin"]))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def condition_color(cond):
    colors = {
        "width_wide": "tab:blue",
        "width_narrow": "tab:orange",
        "length_long": "tab:green",
    }
    return colors.get(str(cond), None)


def plot_metric_bin_size_lines(metric_bin_df, ycol, secol, ylabel, title, filename):
    if metric_bin_df.empty:
        return

    fig, axes = plt.subplots(1, len(BIN_SIZES), figsize=(17, 4.8), sharey=False)
    if len(BIN_SIZES) == 1:
        axes = [axes]

    handles = []
    labels = []

    for ax, bin_size in zip(axes, BIN_SIZES):
        tmp = metric_bin_df[metric_bin_df["binSize"] == bin_size].copy()
        for cond in COND_ORDER:
            sub = tmp[(tmp["condition"].astype(str) == cond) & tmp[ycol].notna()].sort_values("binCenter")
            if sub.empty:
                continue
            line = ax.errorbar(
                sub["binCenter"],
                sub[ycol],
                yerr=sub[secol] if secol in sub.columns else None,
                marker="o",
                linewidth=1.8,
                capsize=2,
                color=condition_color(cond),
                label=make_condition_metric_label(cond),
            )
            if make_condition_metric_label(cond) not in labels:
                handles.append(line)
                labels.append(make_condition_metric_label(cond))

        ax.set_title(f"{bin_size}-trial bins")
        ax.set_xlabel("Within-condition Core attempt number N")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.suptitle(title)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
        fig.subplots_adjust(bottom=0.26, top=0.84)
    else:
        fig.tight_layout()
    save_png_figure(fig, OUT_DIR / filename, dpi=220)
    plt.close(fig)


def plot_mt_tpe_100trial_average(metric_bin_df):
    tmp = metric_bin_df[metric_bin_df["binSize"] == 100].copy()
    if tmp.empty:
        return

    export_cols = [
        "condition", "condition_label", "binStart", "binEnd", "binCenter",
        "MT_n", "MT_mean", "MT_se", "TPe_n", "TPe_mean", "TPe_se",
    ]
    tmp[export_cols].to_csv(OUT_DIR / "01_MT_TPe_100trial_average.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    specs = [
        ("MT_mean", "MT_se", "Mean Core MT [s]", "MT 100-trial average"),
        ("TPe_mean", "TPe_se", "Mean Core TPe [1/s]", "TPe 100-trial average"),
    ]

    handles = []
    labels = []
    for ax, (ycol, secol, ylabel, title) in zip(axes, specs):
        for cond in COND_ORDER:
            sub = tmp[(tmp["condition"].astype(str) == cond) & tmp[ycol].notna()].sort_values("binCenter")
            if sub.empty:
                continue
            line = ax.errorbar(
                sub["binCenter"],
                sub[ycol],
                yerr=sub[secol],
                marker="o",
                linewidth=2,
                capsize=2,
                color=condition_color(cond),
                label=make_condition_metric_label(cond),
            )
            if make_condition_metric_label(cond) not in labels:
                handles.append(line)
                labels.append(make_condition_metric_label(cond))
        ax.set_title(title)
        ax.set_xlabel("Within-condition Core attempt number N")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Core MT and TPe by 100-attempt bins")
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
        fig.subplots_adjust(bottom=0.25, top=0.84)
    save_png_figure(fig, OUT_DIR / "01_MT_TPe_100trial_average.png", dpi=220)
    plt.close(fig)


def plot_combined_metrics_by_bin_size(metric_bin_df):
    if metric_bin_df.empty:
        return

    metric_bin_df.to_csv(OUT_DIR / "03_metrics_bin_summary.csv", index=False, encoding="utf-8-sig")

    specs = [
        ("MT_mean", "MT_se", "Mean Core MT [s]", "MT"),
        ("TPe_mean", "TPe_se", "Mean Core TPe [1/s]", "TPe"),
        ("ER_mean", "ER_se", "Core error rate [%]", "ER"),
    ]

    for bin_size in BIN_SIZES:
        tmp = metric_bin_df[metric_bin_df["binSize"] == bin_size].copy()
        if tmp.empty:
            continue
        tmp.to_csv(OUT_DIR / f"06_combined_metrics_bin{bin_size}.csv", index=False, encoding="utf-8-sig")

        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True)
        handles = []
        labels = []

        for ax, (ycol, secol, ylabel, title) in zip(axes, specs):
            for cond in COND_ORDER:
                sub = tmp[(tmp["condition"].astype(str) == cond) & tmp[ycol].notna()].sort_values("binCenter")
                if sub.empty:
                    continue
                line = ax.errorbar(
                    sub["binCenter"],
                    sub[ycol],
                    yerr=sub[secol],
                    marker="o",
                    linewidth=1.8,
                    capsize=2,
                    color=condition_color(cond),
                    label=make_condition_metric_label(cond),
                )
                if make_condition_metric_label(cond) not in labels:
                    handles.append(line)
                    labels.append(make_condition_metric_label(cond))
            ax.set_title(title)
            ax.set_xlabel("Within-condition Core attempt number N")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)

        fig.suptitle(f"Core MT, TPe, and ER ({bin_size}-attempt bins)")
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
            fig.subplots_adjust(bottom=0.25, top=0.84)
        save_png_figure(fig, OUT_DIR / f"06_combined_metrics_bin{bin_size}.png", dpi=220)
        plt.close(fig)


def plot_same_id_hypothesis_dashboard(metric_bin_df, bin_size=HYPOTHESIS_BIN_SIZE):
    tmp = metric_bin_df[metric_bin_df["binSize"] == bin_size].copy()
    if tmp.empty:
        return

    present = set(tmp["condition"].dropna().astype(str))
    same_id_pairs = [
        (label, conds) for label, conds in SAME_ID_PAIRS
        if all(cond in present for cond in conds)
    ]
    if not same_id_pairs:
        return
    specs = [
        ("MT_mean", "MT_se", "Mean Core MT [s]", "MT (lower is faster)"),
        ("TPe_mean", "TPe_se", "Mean Core TPe [1/s]", "TPe (higher is better)"),
        ("ER_mean", "ER_se", "Core error rate [%]", "ER (lower is more stable)"),
    ]

    fig, axes = plt.subplots(
        len(same_id_pairs), len(specs), figsize=(17, 4.8 * len(same_id_pairs)),
        sharex=True, squeeze=False,
    )

    for row_idx, (row_title, conds) in enumerate(same_id_pairs):
        for col_idx, (ycol, secol, ylabel, title) in enumerate(specs):
            ax = axes[row_idx, col_idx]
            for cond in conds:
                sub = tmp[(tmp["condition"].astype(str) == cond) & tmp[ycol].notna()].sort_values("binCenter")
                if sub.empty:
                    continue
                ax.errorbar(
                    sub["binCenter"],
                    sub[ycol],
                    yerr=sub[secol],
                    marker="o",
                    linewidth=2,
                    capsize=2,
                    color=condition_color(cond),
                    label=make_condition_metric_label(cond),
                )
            ax.set_title(title if row_idx == 0 else "")
            ax.set_xlabel("Within-condition Core attempt number N")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            if col_idx == 0:
                ax.text(
                    -0.20,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                )
            if row_idx == 0 and col_idx == len(specs) - 1:
                ax.legend(loc="upper right", fontsize=8)
            elif row_idx == 1 and col_idx == len(specs) - 1:
                ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Core: same-ID hypothesis dashboard ({bin_size}-attempt bins)")
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.94))
    save_png_figure(fig, OUT_DIR / "07_same_ID_hypothesis_dashboard.png", dpi=230)
    plt.close(fig)


def plot_hypothesis_contrasts(metric_bin_df, bin_size=HYPOTHESIS_BIN_SIZE):
    tmp = metric_bin_df[metric_bin_df["binSize"] == bin_size].copy()
    if tmp.empty:
        return

    contrast_defs = [(label, left, right) for _, label, left, right in CONTRAST_DEFS]
    metric_defs = [
        ("MT_mean", "Core MT difference [s]", "MT diff (negative means left is faster)"),
        ("TPe_mean", "Core TPe difference [1/s]", "TPe diff (positive means left is better)"),
        ("ER_mean", "Core ER difference [percentage points]", "ER diff (negative means left is more stable)"),
    ]

    contrast_rows = []
    for ycol, _, _ in metric_defs:
        pivot = tmp.pivot_table(index="binCenter", columns="condition", values=ycol, observed=True)
        for contrast_name, left, right in contrast_defs:
            if left not in pivot.columns or right not in pivot.columns:
                continue
            diff = pivot[left] - pivot[right]
            for bin_center, value in diff.dropna().items():
                contrast_rows.append({
                    "binSize": bin_size,
                    "binCenter": bin_center,
                    "metric": ycol,
                    "contrast": contrast_name,
                    "left_condition": left,
                    "right_condition": right,
                    "difference_left_minus_right": value,
                })

    contrast_df_local = pd.DataFrame(contrast_rows)
    if contrast_df_local.empty:
        return
    contrast_df_local.to_csv(OUT_DIR / "08_hypothesis_contrasts_over_trials.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True)
    for ax, (ycol, ylabel, title) in zip(axes, metric_defs):
        sub_metric = contrast_df_local[contrast_df_local["metric"] == ycol]
        for contrast_name in sub_metric["contrast"].dropna().unique():
            sub = sub_metric[sub_metric["contrast"] == contrast_name].sort_values("binCenter")
            ax.plot(
                sub["binCenter"],
                sub["difference_left_minus_right"],
                marker="o",
                linewidth=2,
                label=contrast_name,
            )
        ax.axhline(0, color="0.2", linewidth=1, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Within-condition Core attempt number N")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Core: hypothesis contrasts ({bin_size}-attempt bins)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_png_figure(fig, OUT_DIR / "08_hypothesis_contrasts_over_trials.png", dpi=230)
    plt.close(fig)


def plot_final_bin_hypothesis_summary(metric_bin_df):
    if metric_bin_df.empty:
        return

    final_rows = []
    for cond in COND_ORDER:
        sub = metric_bin_df[
            (metric_bin_df["binSize"] == 100)
            & (metric_bin_df["condition"].astype(str) == cond)
        ].sort_values("binCenter")
        if sub.empty:
            continue
        row = sub.iloc[-1]
        final_rows.append({
            "condition": cond,
            "condition_label": make_condition_metric_label(cond),
            "ID_group": COND_ID_GROUP.get(cond),
            "binStart": row["binStart"],
            "binEnd": row["binEnd"],
            "n_trials": row["n_trials"],
            "MT_mean": row.get("MT_mean", np.nan),
            "TPe_mean": row.get("TPe_mean", np.nan),
            "ER_mean": row.get("ER_mean", np.nan),
        })

    final_df = pd.DataFrame(final_rows)
    if final_df.empty:
        return
    final_df.to_csv(OUT_DIR / "09_hypothesis_final_100trial_summary.csv", index=False, encoding="utf-8-sig")

    specs = [
        ("MT_mean", "Mean Core MT [s]", "Final bin (up to 100 trials): MT"),
        ("TPe_mean", "Mean Core TPe [1/s]", "Final bin (up to 100 trials): TPe"),
        ("ER_mean", "Core error rate [%]", "Final bin (up to 100 trials): ER"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    x = np.arange(len(final_df))
    colors = [condition_color(cond) for cond in final_df["condition"]]
    for ax, (ycol, ylabel, title) in zip(axes, specs):
        ax.bar(x, final_df[ycol], color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(final_df["condition_label"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Core hypothesis check: final available bin (100-attempt bins)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_png_figure(fig, OUT_DIR / "09_hypothesis_final_100trial_summary.png", dpi=230)
    plt.close(fig)


def condition_number_label(cond):
    cond = str(cond)
    return f"Condition {COND_NUMBER.get(cond, '?')}: {make_condition_metric_label(cond)}"


def build_condition_power_law_fits(data):
    fit_rows = []
    fit_map = {}

    for cond in COND_ORDER:
        cond_data = data[data["condition"].astype(str) == cond].copy()
        if cond_data.empty:
            continue

        metric_sources = [
            (
                "MT",
                "Core MT [s]",
                cond_data[cond_data["success_for_mt"] & cond_data["MT"].notna()][["N", "MT"]],
                "MT",
                "N",
                10,
            ),
            (
                "TPe",
                "Core TPe [1/s]",
                cond_data[
                    cond_data["success_clean_for_tpe"]
                    & cond_data["TPe_clean"].notna()
                ][["N", "TPe_clean"]],
                "TPe_clean",
                "N",
                10,
            ),
        ]

        er_fit = (
            cond_data
            .groupby("block20", observed=True)
            .agg(
                N_mid=("N", "mean"),
                ER_percent=("error_counted", lambda x: x.mean() * 100),
            )
            .reset_index()
        )
        er_fit["ER_percent_for_fit"] = er_fit["ER_percent"].clip(lower=1e-4)
        metric_sources.append(
            (
                "ER",
                "Core ER [%]",
                er_fit[["N_mid", "ER_percent_for_fit"]],
                "ER_percent_for_fit",
                "N_mid",
                6,
            )
        )

        for metric, ylabel, source, ycol, ncol, min_points in metric_sources:
            res = fit_power_law_signed_a(
                source,
                ycol=ycol,
                ncol=ncol,
                min_points=min_points,
            )
            row = {
                "condition_number": COND_NUMBER.get(cond),
                "condition": cond,
                "condition_label": make_condition_metric_label(cond),
                "ID_group": COND_ID_GROUP.get(cond),
                "metric": metric,
                "ylabel": ylabel,
                **res,
            }
            fit_rows.append(row)
            fit_map[(cond, metric)] = row

    fit_condition_df = pd.DataFrame(fit_rows)
    fit_condition_df.to_csv(
        OUT_DIR / "10_condition_power_law_fit_params.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return fit_condition_df, fit_map


def plot_power_law_curve(ax, fit_map, cond, metric, x_min, x_max, color=None, label=None):
    fit = fit_map.get((cond, metric))
    if not fit or fit.get("status") != "ok":
        return
    a = fit.get("a", np.nan)
    b = fit.get("b", np.nan)
    c = fit.get("c", np.nan)
    if pd.isna(a) or pd.isna(b) or pd.isna(c):
        return

    x = np.linspace(max(1, x_min), max(x_min + 1, x_max), 240)
    y = power_law(x, a, b, c)
    y = np.where(np.isfinite(y), y, np.nan)
    ax.plot(
        x,
        y,
        linestyle="--",
        linewidth=2,
        color=color,
        label=label,
    )


def plot_condition_individual_1trial_100line_fit(data, metric_bin_df, fit_map):
    for cond in COND_ORDER:
        cond_data = data[data["condition"].astype(str) == cond].copy()
        if cond_data.empty:
            continue

        cond_100 = metric_bin_df[
            (metric_bin_df["binSize"] == 100)
            & (metric_bin_df["condition"].astype(str) == cond)
        ].copy().sort_values("binCenter")

        color = condition_color(cond)
        x_max = max(cond_data["N"].max(), 1)
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

        metric_specs = [
            (
                "MT",
                "Core MT [s]",
                cond_data[cond_data["success_for_mt"] & cond_data["MT"].notna()],
                "MT",
                cond_100,
                "MT_mean",
                "MT_se",
            ),
            (
                "TPe",
                "Core TPe [1/s]",
                cond_data[
                    cond_data["success_clean_for_tpe"]
                    & cond_data["TPe_clean"].notna()
                ],
                "TPe_clean",
                cond_100,
                "TPe_mean",
                "TPe_se",
            ),
            (
                "ER",
                "Core ER [%]",
                cond_data.assign(ER_trial=cond_data["error_counted"].astype(float) * 100),
                "ER_trial",
                cond_100,
                "ER_mean",
                "ER_se",
            ),
        ]

        for ax, (metric, ylabel, raw, raw_col, binned, mean_col, se_col) in zip(axes, metric_specs):
            if not raw.empty:
                ax.scatter(
                    raw["N"],
                    raw[raw_col],
                    s=12,
                    alpha=0.22 if metric != "ER" else 0.16,
                    color=color,
                    label="Single trials",
                )

            binned_valid = binned[binned[mean_col].notna()] if mean_col in binned.columns else pd.DataFrame()
            if not binned_valid.empty:
                ax.errorbar(
                    binned_valid["binCenter"],
                    binned_valid[mean_col],
                    yerr=binned_valid[se_col] if se_col in binned_valid.columns else None,
                    marker="o",
                    linewidth=2.5,
                    capsize=3,
                    color="black",
                    label="100-trial average",
                )

            plot_power_law_curve(
                ax,
                fit_map,
                cond,
                metric,
                1,
                x_max,
                color="crimson",
                label="Nonlinear fit aN^{-b}+c",
            )
            ax.set_ylabel(ylabel)
            ax.set_title(metric)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)

        axes[-1].set_xlabel("Within-condition Core attempt number N")
        fig.suptitle(f"{condition_number_label(cond)}: Core metrics, single attempts, 100-attempt average, nonlinear fit")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save_png_figure(fig,
            OUT_DIR / f"10_condition_{COND_NUMBER.get(cond)}_MT_TPe_ER_1trial_100trial_fit.png",
            dpi=230,
        )
        plt.close(fig)


SINGLE_TRIAL_METRIC_SPECS = [
    ("MT", "MT_single", "Core MT [s]", "MT per trial"),
    ("TPe", "TPe_single", "Core TPe [1/s]", "TPe per trial"),
    ("ER", "ER_single", "Core ER [%]", "ER per trial"),
]


def build_single_trial_metric_frame(cond_data, cond):
    mt_mask = cond_data["success_for_mt"] & cond_data["MT"].notna()
    tpe_mask = cond_data["success_clean_for_tpe"] & cond_data["TPe_clean"].notna()

    return pd.DataFrame({
        "condition": cond_data["condition"].astype(str).to_numpy(),
        "condition_label": make_condition_metric_label(cond),
        "N": cond_data["N"].to_numpy(),
        "MT_single": np.where(mt_mask, cond_data["MT"], np.nan),
        "TPe_single": np.where(tpe_mask, cond_data["TPe_clean"], np.nan),
        "ER_single": cond_data["error_counted"].astype(float).to_numpy() * 100,
        "MT_included": mt_mask.to_numpy(),
        "TPe_included": tpe_mask.to_numpy(),
        "MT_omission_reason": metric_omission_reasons(cond_data, mt_mask, "MT"),
        "TPe_omission_reason": metric_omission_reasons(cond_data, tpe_mask, "TPe"),
    })


def metric_omission_reasons(cond_data, include_mask, metric):
    reasons = pd.Series("included", index=cond_data.index, dtype="object")
    excluded = ~include_mask

    if metric == "MT":
        reasons.loc[excluded & cond_data["MT"].isna()] = "mt_missing"
        reasons.loc[excluded & ~cond_data["performance_trial_before_outlier"]] = "not_performance_trial"
        reasons.loc[excluded & cond_data["exclude_by_deviation_ratio"]] = "deviation_ratio_excluded"
        reasons.loc[excluded & cond_data["is_mt_outlier"]] = "mt_outlier"
    elif metric == "TPe":
        reasons.loc[excluded & cond_data["is_mt_outlier"]] = "mt_outlier"
        reasons.loc[excluded & ~cond_data["performance_trial_before_outlier"]] = "not_performance_trial"
        reasons.loc[excluded & cond_data["exclude_by_deviation_ratio"]] = "deviation_ratio_excluded"
        reasons.loc[
            excluded
            & cond_data["success_for_mt"]
            & ~cond_data["success_clean_for_tpe"]
        ] = "deviated_excluded"
        reasons.loc[
            excluded
            & cond_data["success_clean_for_tpe"]
            & cond_data["TPe_clean"].isna()
        ] = "tpe_source_missing"

    reasons.loc[excluded & ~cond_data["core_completed"]] = "core_incomplete"
    if not INCLUDE_DEVIATED_TRIALS:
        reasons.loc[excluded & cond_data["core_completed"] & cond_data["core_deviated"]] = "core_deviation_excluded"
    reasons.loc[excluded & reasons.eq("included")] = "excluded"
    return reasons.to_numpy()


def fit_linear_trend(single, ycol):
    valid = single[["N", ycol]].dropna()
    result = {
        "status": "insufficient_points",
        "slope": np.nan,
        "intercept": np.nan,
        "r2": np.nan,
        "n_points": len(valid),
    }
    if len(valid) < 2 or valid["N"].nunique() < 2:
        return result, None, None

    slope, intercept = np.polyfit(valid["N"], valid[ycol], deg=1)
    y_pred = slope * valid["N"] + intercept
    ss_res = float(np.sum((valid[ycol] - y_pred) ** 2))
    ss_tot = float(np.sum((valid[ycol] - valid[ycol].mean()) ** 2))

    result.update({
        "status": "ok",
        "slope": slope,
        "intercept": intercept,
        "r2": np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot),
    })

    x_fit = np.linspace(valid["N"].min(), valid["N"].max(), 200)
    y_fit = slope * x_fit + intercept
    return result, x_fit, y_fit


def plot_single_trial_axis(ax, single, ycol, ylabel, title, color):
    ax.scatter(
        single["N"],
        single[ycol],
        s=14,
        color=color,
        alpha=0.85,
        label="Single trials",
    )

    fit, x_fit, y_fit = fit_linear_trend(single, ycol)
    if fit["status"] == "ok":
        ax.plot(
            x_fit,
            y_fit,
            color="crimson",
            linestyle="--",
            linewidth=2,
            label="Linear regression",
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    if ycol == "ER_single":
        ax.set_ylim(-5, 105)
    return fit


def write_single_trial_outputs(single_frames, fit_rows):
    if not single_frames:
        return

    single_df = pd.concat(single_frames, ignore_index=True)
    single_df.to_csv(
        OUT_DIR / "12_condition_single_trial_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_single_trial_omission_summary(single_df).to_csv(
        OUT_DIR / "12_condition_single_trial_omission_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if fit_rows:
        pd.DataFrame(fit_rows).to_csv(
            OUT_DIR / "12_condition_single_trial_linear_fit_params.csv",
            index=False,
            encoding="utf-8-sig",
        )


def build_single_trial_omission_summary(single_df):
    rows = []
    reason_cols = [
        ("MT", "MT_omission_reason"),
        ("TPe", "TPe_omission_reason"),
    ]

    for (cond, label), sub in single_df.groupby(["condition", "condition_label"], observed=True):
        for metric, reason_col in reason_cols:
            counts = sub[reason_col].value_counts(dropna=False)
            for reason, count in counts.items():
                rows.append({
                    "condition_number": COND_NUMBER.get(cond),
                    "condition": cond,
                    "condition_label": label,
                    "metric": metric,
                    "reason": reason,
                    "n_trials": int(count),
                    "percent": float(count / len(sub) * 100) if len(sub) else np.nan,
                })

    return pd.DataFrame(rows).sort_values(["condition_number", "metric", "reason"])


def plot_condition_single_trial_metrics(data):
    single_frames = []
    fit_rows = []

    for cond in COND_ORDER:
        cond_data = data[data["condition"].astype(str) == cond].copy().sort_values("N")
        if cond_data.empty:
            continue

        single = build_single_trial_metric_frame(cond_data, cond)
        single_frames.append(single)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)
        for ax, (metric, ycol, ylabel, title) in zip(axes, SINGLE_TRIAL_METRIC_SPECS):
            fit = plot_single_trial_axis(
                ax,
                single,
                ycol=ycol,
                ylabel=ylabel,
                title=title,
                color=condition_color(cond),
            )
            fit_rows.append({
                "condition_number": COND_NUMBER.get(cond),
                "condition": cond,
                "condition_label": make_condition_metric_label(cond),
                "metric": metric,
                **fit,
            })

        axes[-1].set_xlabel("Within-condition Core attempt number N")
        fig.suptitle(f"{condition_number_label(cond)}: Core MT, TPe, and ER per attempt with linear regression")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save_png_figure(fig,
            OUT_DIR / f"12_condition_{COND_NUMBER.get(cond)}_MT_TPe_ER_single_trial.png",
            dpi=230,
        )
        plt.close(fig)

    write_single_trial_outputs(single_frames, fit_rows)


def plot_same_id_mt_tpe_evaluation(data, metric_bin_df, fit_map, fit_condition_df):
    present = set(data["condition"].dropna().astype(str))
    pairs = [
        (label, conds) for label, conds in SAME_ID_PAIRS
        if all(cond in present for cond in conds)
    ]
    if not pairs:
        return
    metrics = [
        ("MT", "Core MT [s]", "MT", "MT_mean", "MT_se", "lower is faster"),
        ("TPe", "Core TPe [1/s]", "TPe_clean", "TPe_mean", "TPe_se", "higher is better"),
    ]

    fig, axes = plt.subplots(
        len(pairs), len(metrics), figsize=(15, 4.8 * len(pairs)),
        sharex=True, squeeze=False,
    )

    for row_idx, (id_group, conds) in enumerate(pairs):
        for col_idx, (metric, ylabel, raw_col, mean_col, se_col, note) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for cond in conds:
                color = condition_color(cond)
                cond_data = data[data["condition"].astype(str) == cond].copy()
                if metric == "MT":
                    raw = cond_data[cond_data["success_for_mt"] & cond_data["MT"].notna()]
                else:
                    raw = cond_data[
                        cond_data["success_clean_for_tpe"]
                        & cond_data["TPe_clean"].notna()
                    ]
                if not raw.empty:
                    ax.scatter(
                        raw["N"],
                        raw[raw_col],
                        s=10,
                        alpha=0.16,
                        color=color,
                    )

                binned = metric_bin_df[
                    (metric_bin_df["binSize"] == 100)
                    & (metric_bin_df["condition"].astype(str) == cond)
                    & metric_bin_df[mean_col].notna()
                ].sort_values("binCenter")
                if not binned.empty:
                    ax.errorbar(
                        binned["binCenter"],
                        binned[mean_col],
                        yerr=binned[se_col],
                        marker="o",
                        linewidth=2.5,
                        capsize=3,
                        color=color,
                        label=f"{make_condition_metric_label(cond)} 100-trial average",
                    )

                plot_power_law_curve(
                    ax,
                    fit_map,
                    cond,
                    metric,
                    1,
                    max(cond_data["N"].max(), 1) if not cond_data.empty else 300,
                    color=color,
                    label=f"{make_condition_metric_label(cond)} nonlinear fit",
                )

            ax.set_title(f"{id_group} {metric}: {note}")
            ax.set_xlabel("Within-condition Core attempt number N")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=7)

    fig.suptitle("Same-ID Core MT and TPe (single attempts + 100-attempt average + nonlinear fit)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_png_figure(fig, OUT_DIR / "11_same_ID_MT_TPe_evaluation_1trial_100trial_fit.png", dpi=230)
    plt.close(fig)

    same_id_fit = fit_condition_df[
        fit_condition_df["metric"].isin(["MT", "TPe"])
        & fit_condition_df["ID_group"].isin([label for label, _ in pairs])
    ].copy()
    same_id_fit.to_csv(
        OUT_DIR / "11_same_ID_MT_TPe_power_law_fit_params.csv",
        index=False,
        encoding="utf-8-sig",
    )


REPORT_METRIC_SPECS = {
    "MT": {
        "value_col": "MT",
        "mask_col": "success_for_mt",
        "ylabel": "Core MT [s]",
        "title": "MT",
        "filename": "02_MT",
    },
    "TPe": {
        "value_col": "TPe_clean",
        "mask_col": "success_clean_for_tpe",
        "ylabel": "Core TPe [1/s]",
        "title": "TPe",
        "filename": "04_TPe",
    },
    "We": {
        "value_col": "We",
        "mask_col": None,
        "ylabel": "Core We [px]",
        "title": "We",
        "filename": "03_We",
    },
    "Ae": {
        "value_col": "Ae",
        "mask_col": None,
        "ylabel": "Core Ae [px]",
        "title": "Ae",
        "filename": "05_Ae",
    },
}


def add_continuous_trial_index(data):
    sort_cols = [
        col for col in [
            "participant",
            "started_at",
            "source_file",
            "totalTrial",
            "trialInPhase",
            "trial_original",
            "condition",
            "N",
        ]
        if col in data.columns
    ]
    out = data.sort_values(sort_cols).copy()
    out["global_N"] = np.arange(1, len(out) + 1)
    out["global_block100"] = ((out["global_N"] - 1) // 100) + 1
    return out


def numeric_or_nan(data, col):
    if col in data.columns:
        return pd.to_numeric(data[col], errors="coerce")
    return pd.Series(np.nan, index=data.index)


def summarize_condition_report(data):
    rows = []
    success_bool = data["core_completed"]
    deviated_bool = data["core_deviated"].astype(float).where(data["core_deviation_known"])

    for cond in COND_ORDER:
        sub = data[data["condition"].astype(str) == cond].copy()
        if sub.empty:
            continue

        success_sub = success_bool.loc[sub.index]
        deviated_sub = deviated_bool.loc[sub.index]
        mt_sub = sub.loc[sub["success_for_mt"], "MT"].dropna()
        tpe_sub = sub.loc[sub["success_clean_for_tpe"], "TPe_clean"].dropna()

        rows.append({
            "condition_number": COND_NUMBER.get(cond),
            "condition": cond,
            "condition_label": make_condition_metric_label(cond),
            "analysis_scope": ANALYSIS_SCOPE,
            "n_trials": len(sub),
            "n_core_completed": int(success_sub.sum()),
            "core_completion_rate_percent": float(success_sub.mean() * 100) if len(sub) else np.nan,
            "n_core_incomplete": int((~success_sub).sum()),
            "n_error_counted": int(sub["error_counted"].sum()),
            "error_rate_percent": float(sub["error_counted"].mean() * 100) if len(sub) else np.nan,
            "n_deviated": int(deviated_sub.sum()),
            "deviation_rate_percent": float(deviated_sub.mean() * 100) if len(sub) else np.nan,
            "n_core_deviation_unknown": int((~sub["core_deviation_known"]).sum()),
            "deviation_count_mean": numeric_or_nan(sub, "core_deviation_count").mean(),
            "deviation_count_sum": numeric_or_nan(sub, "core_deviation_count").sum(min_count=1),
            "deviation_time_ms_mean": numeric_or_nan(sub, "core_deviation_total_ms").mean(),
            "deviation_time_ms_sum": numeric_or_nan(sub, "core_deviation_total_ms").sum(min_count=1),
            "deviation_time_ratio_mean": numeric_or_nan(sub, "deviation_time_ratio").mean(),
            "n_excluded_by_deviation_ratio": int(sub["exclude_by_deviation_ratio"].sum()),
            "deviation_ratio_exclusion_threshold": DEVIATION_EXCLUSION_RATIO,
            "max_deviation_px_mean": numeric_or_nan(sub, "core_max_deviation_px").mean(),
            "max_deviation_px_max": numeric_or_nan(sub, "core_max_deviation_px").max(),
            "n_mt_analysis": len(mt_sub),
            "mt_mean": mt_sub.mean(),
            "mt_sd": mt_sub.std(),
            "mt_median": mt_sub.median(),
            "n_tpe_analysis": len(tpe_sub),
            "tpe_mean": tpe_sub.mean(),
            "tpe_sd": tpe_sub.std(),
            "tpe_median": tpe_sub.median(),
            "we_mean": numeric_or_nan(sub, "We").mean(),
            "ae_mean": numeric_or_nan(sub, "Ae").mean(),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(CONDITION_REPORT_DIR / "01_condition_basic_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def write_error_and_deviation_exclusion_reports(data):
    error_counts = (
        data.groupby(["condition", "error_text"], observed=True)
        .size()
        .reset_index(name="n_trials")
    )
    condition_totals = data.groupby("condition", observed=True).size().rename("condition_total")
    error_counts = error_counts.merge(condition_totals, on="condition", how="left")
    error_counts["condition_number"] = error_counts["condition"].map(COND_NUMBER)
    error_counts["condition_label"] = error_counts["condition"].map(make_condition_metric_label)
    error_counts["percent_in_condition"] = error_counts["n_trials"] / error_counts["condition_total"] * 100
    error_counts = error_counts[
        [
            "condition_number",
            "condition",
            "condition_label",
            "error_text",
            "n_trials",
            "condition_total",
            "percent_in_condition",
        ]
    ].sort_values(["condition_number", "error_text"])
    error_counts.to_csv(
        CONDITION_REPORT_DIR / "01_error_reason_counts_by_condition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rows = []
    for cond in COND_ORDER:
        sub = data[data["condition"].astype(str) == cond].copy()
        if sub.empty:
            continue
        rows.append({
            "condition_number": COND_NUMBER.get(cond),
            "condition": cond,
            "condition_label": make_condition_metric_label(cond),
            "n_trials": len(sub),
            "n_deviation_error_text": int((sub["core_deviated"] & ~sub["core_excessive_deviation"]).sum()),
            "n_excessive_deviation_error_text": int(sub["core_excessive_deviation"].sum()),
            "n_deviated_true": int(sub["core_deviated"].sum()),
            "n_core_incomplete": int((~sub["core_completed"]).sum()),
            "n_core_deviation_unknown": int((~sub["core_deviation_known"]).sum()),
            "n_excluded_by_deviation_ratio": int(sub["exclude_by_deviation_ratio"].sum()),
            "deviation_ratio_exclusion_threshold": DEVIATION_EXCLUSION_RATIO,
            "max_deviation_time_ratio": numeric_or_nan(sub, "deviation_time_ratio").max(),
            "mean_deviation_time_ratio": numeric_or_nan(sub, "deviation_time_ratio").mean(),
        })

    pd.DataFrame(rows).to_csv(
        CONDITION_REPORT_DIR / "01_deviation_exclusion_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_condition_report_summary(summary):
    if summary.empty:
        return

    labels = summary["condition_label"]
    x = np.arange(len(summary))
    colors = [condition_color(cond) for cond in summary["condition"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    rate_specs = [
        ("core_completion_rate_percent", "Core completion rate [%]"),
        ("error_rate_percent", "Core error rate [%]"),
        ("deviation_rate_percent", "Core deviation rate [%]"),
    ]
    for ax, (col, title) in zip(axes, rate_specs):
        ax.bar(x, summary[col], color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Core interval: completion, error, and deviation rates")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_png_figure(fig, CONDITION_REPORT_DIR / "01_condition_rates.png", dpi=230)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    mean_specs = [
        ("mt_mean", "Mean Core MT [s]"),
        ("tpe_mean", "Mean Core TPe [1/s]"),
    ]
    for ax, (col, title) in zip(axes, mean_specs):
        ax.bar(x, summary[col], color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Condition summary: mean Core MT and TPe")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_png_figure(fig, CONDITION_REPORT_DIR / "01_condition_mean_MT_TPe.png", dpi=230)
    plt.close(fig)


def build_report_metric_frame(data, metric):
    spec = REPORT_METRIC_SPECS[metric]
    value = numeric_or_nan(data, spec["value_col"])
    if spec["mask_col"] is not None:
        value = value.where(data[spec["mask_col"]])

    out = data[["condition", "N", "global_N", "global_block100"]].copy()
    out["condition"] = out["condition"].astype(str)
    out["condition_label"] = out["condition"].map(make_condition_metric_label)
    out["metric"] = metric
    out["value"] = value
    return out


def fit_linear_xy(x, y):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    result = {
        "status": "insufficient_points",
        "slope": np.nan,
        "intercept": np.nan,
        "r2": np.nan,
        "n_points": len(d),
    }
    if len(d) < 2 or d["x"].nunique() < 2:
        return result, None, None

    slope, intercept = np.polyfit(d["x"], d["y"], deg=1)
    y_pred = slope * d["x"] + intercept
    ss_res = float(np.sum((d["y"] - y_pred) ** 2))
    ss_tot = float(np.sum((d["y"] - d["y"].mean()) ** 2))
    result.update({
        "status": "ok",
        "slope": slope,
        "intercept": intercept,
        "r2": np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot),
    })

    x_fit = np.linspace(d["x"].min(), d["x"].max(), 200)
    y_fit = slope * x_fit + intercept
    return result, x_fit, y_fit


def plot_metric_by_condition(report_metric_df, metric):
    spec = REPORT_METRIC_SPECS[metric]
    fit_rows = []

    for cond in COND_ORDER:
        sub = report_metric_df[report_metric_df["condition"] == cond].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 4.8))
        color = condition_color(cond)
        ax.scatter(sub["N"], sub["value"], s=14, alpha=0.75, color=color, label="Trials")
        fit, x_fit, y_fit = fit_linear_xy(sub["N"], sub["value"])
        if fit["status"] == "ok":
            ax.plot(x_fit, y_fit, color="crimson", linestyle="--", linewidth=2, label="Linear regression")

        fit_rows.append({
            "metric": metric,
            "fit_scope": "condition",
            "condition_number": COND_NUMBER.get(cond),
            "condition": cond,
            "condition_label": make_condition_metric_label(cond),
            **fit,
        })
        ax.set_title(f"{make_condition_metric_label(cond)}: {spec['title']} trend")
        ax.set_xlabel("Within-condition Core attempt number N")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        save_png_figure(fig, CONDITION_REPORT_DIR / f"{spec['filename']}_condition_{COND_NUMBER.get(cond)}_trend.png", dpi=230)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for cond in COND_ORDER:
        sub = report_metric_df[report_metric_df["condition"] == cond].copy()
        if sub.empty:
            continue
        ax.scatter(
            sub["N"],
            sub["value"],
            s=12,
            alpha=0.55,
            color=condition_color(cond),
            label=make_condition_metric_label(cond),
        )
    ax.set_title(f"{spec['title']} trends by condition")
    ax.set_xlabel("Within-condition Core attempt number N")
    ax.set_ylabel(spec["ylabel"])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_png_figure(fig, CONDITION_REPORT_DIR / f"{spec['filename']}_all_conditions_trend.png", dpi=230)
    plt.close(fig)

    mean_df = (
        report_metric_df.dropna(subset=["value"])
        .groupby(["condition", "condition_label"], observed=True)["value"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    mean_df["condition_number"] = mean_df["condition"].map(COND_NUMBER)
    mean_df.sort_values("condition_number").to_csv(
        CONDITION_REPORT_DIR / f"{spec['filename']}_mean_by_condition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not mean_df.empty:
        mean_df = mean_df.sort_values("condition_number")
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(
            np.arange(len(mean_df)),
            mean_df["mean"],
            color=[condition_color(cond) for cond in mean_df["condition"]],
            alpha=0.85,
        )
        ax.set_xticks(np.arange(len(mean_df)))
        ax.set_xticklabels(mean_df["condition_label"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(spec["ylabel"])
        ax.set_title(f"Mean {spec['title']} by condition")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        save_png_figure(fig, CONDITION_REPORT_DIR / f"{spec['filename']}_mean_by_condition.png", dpi=230)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for cond in COND_ORDER:
        sub = report_metric_df[report_metric_df["condition"] == cond]
        if sub.empty:
            continue
        ax.scatter(
            sub["global_N"],
            sub["value"],
            s=12,
            alpha=0.55,
            color=condition_color(cond),
            label=make_condition_metric_label(cond),
        )

    fit, x_fit, y_fit = fit_linear_xy(report_metric_df["global_N"], report_metric_df["value"])
    if fit["status"] == "ok":
        ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=2, label="Overall linear regression")
    fit_rows.append({
        "metric": metric,
        "fit_scope": "continuous_all_trials",
        "condition_number": np.nan,
        "condition": "all",
        "condition_label": "All conditions",
        **fit,
    })

    ax.set_title(f"{spec['title']} over all trials")
    ax.set_xlabel("Continuous Core attempt number")
    ax.set_ylabel(spec["ylabel"])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_png_figure(fig, CONDITION_REPORT_DIR / f"{spec['filename']}_continuous_all_trials.png", dpi=230)
    plt.close(fig)

    return fit_rows


def minmax_normalize(series):
    series = pd.to_numeric(series, errors="coerce")
    valid = series.dropna()
    if valid.empty or valid.max() == valid.min():
        return pd.Series(np.nan, index=series.index)
    return (series - valid.min()) / (valid.max() - valid.min())


def plot_we_ae_tpe_relationships(report_data):
    source = report_data.assign(MT=report_data["MT"].where(report_data["success_for_mt"]))
    block = (
        source.groupby("global_block100", observed=True)
        .agg(
            global_N=("global_N", "mean"),
            We_mean=("We", "mean"),
            Ae_mean=("Ae", "mean"),
            MT_mean=("MT", "mean"),
            TPe_mean=("TPe_clean", "mean"),
        )
        .reset_index()
    )
    block.to_csv(CONDITION_REPORT_DIR / "05_We_Ae_MT_TPe_100trial_blocks.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for ax, ycol, ylabel, title in [
        (axes[0], "We_mean", "Mean Core We [px]", "We by 100-trial blocks"),
        (axes[1], "Ae_mean", "Mean Core Ae [px]", "Ae by 100-trial blocks"),
    ]:
        ax.plot(block["global_N"], block[ycol], marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Continuous Core attempt number")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_png_figure(fig, CONDITION_REPORT_DIR / "05_We_Ae_100trial_blocks.png", dpi=230)
    plt.close(fig)

    normalized = pd.DataFrame({
        "global_N": report_data["global_N"],
        "MT_norm": minmax_normalize(report_data["MT"].where(report_data["success_for_mt"])),
        "We_norm": minmax_normalize(report_data["We"]),
        "Ae_norm": minmax_normalize(report_data["Ae"]),
        "TPe_norm": minmax_normalize(report_data["TPe_clean"]),
    })
    normalized.to_csv(CONDITION_REPORT_DIR / "05_normalized_MT_We_Ae_TPe.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 5))
    for col, label in [
        ("MT_norm", "MT"),
        ("We_norm", "We"),
        ("Ae_norm", "Ae"),
        ("TPe_norm", "TPe"),
    ]:
        ax.plot(normalized["global_N"], normalized[col], linewidth=1.4, alpha=0.8, label=label)
    ax.set_title("Normalized Core MT, We, Ae, and TPe over Core attempts")
    ax.set_xlabel("Continuous Core attempt number")
    ax.set_ylabel("Min-max normalized value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_png_figure(fig, CONDITION_REPORT_DIR / "05_normalized_MT_We_Ae_TPe.png", dpi=230)
    plt.close(fig)

    scatter_data = report_data[report_data["TPe_clean"].notna()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, xcol, xlabel, title in [
        (axes[0], "We", "Core We [px]", "TPe vs We"),
        (axes[1], "Ae", "Core Ae [px]", "TPe vs Ae"),
    ]:
        for cond in COND_ORDER:
            sub = scatter_data[scatter_data["condition"].astype(str) == cond]
            if sub.empty:
                continue
            ax.scatter(
                sub[xcol],
                sub["TPe_clean"],
                s=18,
                alpha=0.6,
                color=condition_color(cond),
                label=make_condition_metric_label(cond),
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Core TPe [1/s]")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_png_figure(fig, CONDITION_REPORT_DIR / "05_TPe_vs_We_Ae_scatter.png", dpi=230)
    plt.close(fig)


def generate_condition_report_outputs(data):
    report_data = add_continuous_trial_index(data)
    summary = summarize_condition_report(report_data)
    write_error_and_deviation_exclusion_reports(report_data)
    plot_condition_report_summary(summary)

    fit_rows = []
    for metric in ["MT", "We", "TPe", "Ae"]:
        metric_df = build_report_metric_frame(report_data, metric)
        metric_df.to_csv(
            CONDITION_REPORT_DIR / f"{REPORT_METRIC_SPECS[metric]['filename']}_trial_values.csv",
            index=False,
            encoding="utf-8-sig",
        )
        fit_rows.extend(plot_metric_by_condition(metric_df, metric))

    plot_we_ae_tpe_relationships(report_data)

    if fit_rows:
        pd.DataFrame(fit_rows).to_csv(
            CONDITION_REPORT_DIR / "00_linear_fit_params.csv",
            index=False,
            encoding="utf-8-sig",
        )


# MT
plot_block_metric(
    df[df["success_for_mt"]],
    ycol="MT",
    ylabel="Core MT, eligible completions [s]",
    filename="learning_curve_MT_success.png"
)

# ER
er_plot_df = (
    df.groupby(["participant", "condition", "block20"], observed=True)
    .agg(
        N=("N", "mean"),
        error_rate=("error_counted", "mean"),
    )
    .reset_index()
)
er_plot_df["N"] = er_plot_df["N"].round().astype(int)
plot_block_metric(
    er_plot_df.rename(columns={"N": "N"}),
    ycol="error_rate",
    ylabel="Core ER [proportion]",
    filename="learning_curve_ER.png"
)

# 1/TPe main analysis
if df["inv_TPe"].notna().sum() > 0:
    plot_block_metric(
        df[df["success_clean_for_tpe"] & df["inv_TPe"].notna()],
        ycol="inv_TPe",
        ylabel="1 / Core TPe, main analysis [s]",
        filename="learning_curve_inv_TPe.png"
    )

# 1/TPe auxiliary analysis
if df["inv_TPe_with_deviation"].notna().sum() > 0:
    plot_block_metric(
        df[df["success_with_deviation_for_tpe"] & df["inv_TPe_with_deviation"].notna()],
        ycol="inv_TPe_with_deviation",
        ylabel="1 / Core TPe, including Core-deviated completions [s]",
        filename="learning_curve_inv_TPe_with_deviation.png"
    )

metric_bin_sizes = sorted(set(BIN_SIZES + [HYPOTHESIS_BIN_SIZE]))
metric_bin_df = build_metric_bin_summary(df, metric_bin_sizes)
if not metric_bin_df.empty:
    condition_fit_df, condition_fit_map = build_condition_power_law_fits(df)
    plot_mt_tpe_100trial_average(metric_bin_df)
    plot_metric_bin_size_lines(
        metric_bin_df,
        ycol="MT_mean",
        secol="MT_se",
        ylabel="Mean Core MT [s]",
        title="Core MT averaged by 10/50/100-attempt bins",
        filename="03_MT_10_50_100_average.png",
    )
    plot_metric_bin_size_lines(
        metric_bin_df,
        ycol="TPe_mean",
        secol="TPe_se",
        ylabel="Mean Core TPe [1/s]",
        title="Core TPe averaged by 10/50/100-attempt bins (eligible Core completions)",
        filename="04_TPe_10_50_100_average.png",
    )
    plot_metric_bin_size_lines(
        metric_bin_df,
        ycol="ER_mean",
        secol="ER_se",
        ylabel="Core error rate [%]",
        title="Core error rate averaged by 10/50/100-attempt bins",
        filename="05_error_rate_10_50_100_average.png",
    )
    plot_combined_metrics_by_bin_size(metric_bin_df)
    plot_same_id_hypothesis_dashboard(metric_bin_df, bin_size=HYPOTHESIS_BIN_SIZE)
    plot_hypothesis_contrasts(metric_bin_df, bin_size=HYPOTHESIS_BIN_SIZE)
    plot_final_bin_hypothesis_summary(metric_bin_df)
    plot_condition_individual_1trial_100line_fit(
        df,
        metric_bin_df,
        condition_fit_map,
    )
    plot_condition_single_trial_metrics(df)
    plot_same_id_mt_tpe_evaluation(
        df,
        metric_bin_df,
        condition_fit_map,
        condition_fit_df,
    )

generate_condition_report_outputs(df)


# ============================================================
# 11. 低IDの限界速度仮説用の出力
# ============================================================

limit_rows = []

d_mt = fit_df[(fit_df["metric"] == "MT_success") & (fit_df["status"] == "ok")].copy()

for cond, sub in d_mt.groupby("condition", observed=True):
    id_group = COND_ID_GROUP.get(str(cond))

    limit_rows.append({
        "condition": cond,
        "condition_label": COND_DISPLAY.get(str(cond), str(cond)),
        "ID_group": id_group,
        "a_mean_improvement_amount": sub["a"].mean(),
        "b_mean_learning_rate": sub["b"].mean(),
        "c_mean_asymptote": sub["c"].mean(),
        "N50_mean": sub["N50"].mean(),
        "N80_mean": sub["N80"].mean(),
        "N90_mean": sub["N90"].mean(),
        "N95_mean": sub["N95"].mean(),
        "N80_within_400_rate": sub["N80_within_400"].mean(),
        "N90_within_400_rate": sub["N90_within_400"].mean(),
        "N95_within_400_rate": sub["N95_within_400"].mean(),
        "n_participants": sub["participant"].nunique(),
    })

limit_df = pd.DataFrame(limit_rows)
limit_df.to_csv(OUT_DIR / "limit_speed_hypothesis_summary.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 12. 結果の簡易表示
# ============================================================

print("\n===== Power Law summary: MT_success =====")
print(
    summary_df[
        (summary_df["metric"] == "MT_success")
        & (summary_df["param"].isin(["a", "b", "c", "N50", "N80", "N90", "N95"]))
    ][["condition_label", "param", "mean", "sd", "n"]]
)

print("\n===== Limit speed hypothesis summary =====")
print(limit_df)

if len(contrast_df) > 0:
    print("\n===== Bootstrap contrasts: MT_success, b =====")
    print(
        contrast_df[
            (contrast_df["metric"] == "MT_success")
            & (contrast_df["param"] == "b")
        ]
    )

print(f"\nOutput complete: {OUT_DIR.resolve()}")
