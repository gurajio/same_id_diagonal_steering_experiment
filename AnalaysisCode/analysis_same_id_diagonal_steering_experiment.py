from pathlib import Path
import argparse
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
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONDITION_REPORT_DIR = OUT_DIR / "condition_report"
CONDITION_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1ファイルだけ分析したい場合はここにファイル名を書く．空文字ならCSV_FILENAMESまたは全CSVを使う．
DEFAULT_CSV_FILE = "same_id_steering_main_999_C1_20260620_193749.csv"

# 分析するCSVファイル名をここで指定できる．空リストの場合はinputCSV内の全CSVを読む．
# コマンドラインでファイル名を指定した場合は，そちらを優先する．
CSV_FILENAMES = [
    "予備実験C1C2のみ.csv",
    "予備実験C3C4のみ.csv",
]

BOOT_N = 5000
BIN_SIZE = 20
BIN_SIZES = [10, 50, 100]
HYPOTHESIS_BIN_SIZE = 20
MAX_TRIAL_N = 400

# MT/TPeの移動平均
MT_ROLLING_WINDOW = 10
TPE_ROLLING_WINDOW = 20

# We = 4.133 * sigma_d
WE_COEFFICIENT = 4.133

# MT/TPe分析に成功試行だけを使うかどうか
USE_SUCCESS_TRIALS_ONLY = True

# 成功試行のMT外れ値を削除するかどうか
DELETE_OUTLIER = True
OUTLIER_SIGMA = 3.0

# 経路逸脱した成功試行をMT/TPe計算に含めるかどうか
# False: success == True かつ deviated == False のみ使う
# True : success == True なら使う
INCLUDE_DEVIATED_TRIALS = True

# 経路はみ出しは基本的に許容するが，試行時間のこの割合以上はみ出していた場合は除外する．
# 現CSVでは「経路距離の何割はみ出したか」は直接ないため，deviationTotalMs / mtMs で近似する．
# コマンドライン引数 --deviation-exclusion-ratio で実行時に変更できる．
DEVIATION_EXCLUSION_RATIO = 0.80

# 日本語フォント．MacならHiragino Sans，WindowsならYu Gothicに変更するとよい．
JAPANESE_FONT = "DejaVu Sans"

COUNT_UNLISTED_COUNTED_ERRORS_AS_ERRORS = True

# 試行としてカウントするエラー
COUNTED_ERROR_KEYWORDS = [
    "out_of_path",
    "off_path",
    "deviation",
    "excessive_deviation",
    "too_far",
    "near_goal_pointerup",
    "release_near_goal",
    "path_error",
    "goal_release",
    "other_error",
    "経路はみ出し",
    "はみ出し",
    "はみ出しすぎ",
    "ゴール周囲",
    "その他エラー",
]

# 試行としてカウントしない操作ミス
EXCLUDE_ERROR_KEYWORDS = [
    "not_started",
    "start_error",
    "mid_release",
    "pointerup_mid_path",
    "release_mid_path",
    "operation_mistake",
    "invalid",
    "window_blur",
    "スタートできていない",
    "経路途中で指を離す",
    "途中で指を離す",
    "その他操作ミス",
    "操作ミス",
]

# エラーなしとして扱う値
NO_ERROR_KEYWORDS = [
    "",
    "none",
    "no_error",
    "success",
    "ok",
    "なし",
    "エラーなし",
]

COND_DISPLAY = {
    "width_wide": "Wide-width A1000 W50 ID20",
    "width_narrow": "Narrow-width A1000 W20 ID50",
    "length_long": "Long-distance A1500 W30 ID50",
    "length_short": "Short-distance A600 W30 ID20",
}

COND_ORDER = [
    "width_wide",
    "width_narrow",
    "length_long",
    "length_short",
]

COND_NUMBER = {
    "width_wide": 1,
    "width_narrow": 2,
    "length_long": 3,
    "length_short": 4,
}

COND_ID_GROUP = {
    "width_wide": "ID20",
    "length_short": "ID20",
    "width_narrow": "ID50",
    "length_long": "ID50",
}


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


def make_condition_metric_label(cond):
    cond = str(cond)
    label = COND_DISPLAY.get(cond, cond)
    if cond == "width_wide":
        return f"{label} / C1"
    if cond == "width_narrow":
        return f"{label} / C2"
    if cond == "length_long":
        return f"{label} / C3"
    if cond == "length_short":
        return f"{label} / C4"
    return label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze CSV files in the inputCSV folder."
    )
    parser.add_argument(
        "csv_filenames",
        nargs="*",
        help="CSV filename(s) to analyze. If omitted, CSV_FILENAMES is used; if empty, all CSV files in inputCSV are used.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ignore CSV_FILENAMES and analyze all CSV files in inputCSV.",
    )
    parser.add_argument(
        "--deviation-exclusion-ratio",
        type=float,
        default=None,
        help=(
            "Override DEVIATION_EXCLUSION_RATIO. "
            "Example: 0.8 excludes trials with deviationTotalMs / mtMs >= 0.8."
        ),
    )
    outlier_group = parser.add_mutually_exclusive_group()
    outlier_group.add_argument(
        "--delete-outlier",
        dest="delete_outlier",
        action="store_true",
        default=None,
        help="Remove MT outliers from successful performance trials.",
    )
    outlier_group.add_argument(
        "--keep-outlier",
        dest="delete_outlier",
        action="store_false",
        help="Keep MT outliers in successful performance trials.",
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
        help="Include successful deviated trials in MT/TPe calculations.",
    )
    deviated_group.add_argument(
        "--exclude-deviated-trials",
        dest="include_deviated_trials",
        action="store_false",
        help="Exclude successful deviated trials from MT/TPe calculations.",
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


def contains_keyword(x, keywords):
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return any(k.lower() in s for k in keywords)


def is_no_error_text(x):
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s in {k.lower() for k in NO_ERROR_KEYWORDS}


def build_performance_trial_mask(df, success_col=None, deviated_col=None):
    """
    MT/TPe分析に使う試行を設定から決める．
    ER計算用のerror_countedとは分けて扱う．
    """
    if USE_SUCCESS_TRIALS_ONLY and success_col is not None:
        mask = to_bool_series(df[success_col]).fillna(False)
    elif USE_SUCCESS_TRIALS_ONLY:
        mask = ~df["error_counted"]
    else:
        mask = df["is_counted_trial"].copy()

    if not INCLUDE_DEVIATED_TRIALS and deviated_col is not None:
        deviated = to_bool_series(df[deviated_col]).fillna(False)
        mask = mask & ~deviated

    if "exclude_by_deviation_ratio" in df.columns:
        mask = mask & ~df["exclude_by_deviation_ratio"].fillna(False)

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


def signed_distance_to_segment(px, py, ax, ay, bx, by):
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay

    seg_len2 = vx * vx + vy * vy
    if seg_len2 == 0:
        return np.nan

    t = (wx * vx + wy * vy) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    qx = ax + t * vx
    qy = ay + t * vy

    dx = px - qx
    dy = py - qy
    dist = float(np.sqrt(dx * dx + dy * dy))

    cross = vx * (py - ay) - vy * (px - ax)
    sign = 1 if cross >= 0 else -1
    return sign * dist


def compute_trajectory_length(traj):
    xs = []
    ys = []
    for p in traj:
        try:
            xs.append(float(p["x"]))
            ys.append(float(p["y"]))
        except Exception:
            continue

    if len(xs) < 2:
        return np.nan

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    return float(np.nansum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)))


def compute_effective_tpe_from_trajectory(df, trajectory_col, mask_col):
    """
    添付コードと同じく，We=4.133*sigma_d, Ae=軌跡長, IDe=Ae/We, TPe=IDe/MTで計算する．
    この実験は斜め直線課題なので，中心線は軌跡の始点-終点の直線として扱う．
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

        traj = parse_trajectory_json(row[trajectory_col])
        if len(traj) < 2:
            continue

        try:
            ax = float(traj[0]["x"])
            ay = float(traj[0]["y"])
            bx = float(traj[-1]["x"])
            by = float(traj[-1]["y"])
        except Exception:
            continue

        deviations = []
        for p in traj:
            try:
                x = float(p["x"])
                y = float(p["y"])
            except Exception:
                continue
            d = signed_distance_to_segment(x, y, ax, ay, bx, by)
            if np.isfinite(d):
                deviations.append(d)

        if len(deviations) < 2:
            continue

        sigma_d = float(np.std(deviations, ddof=1))
        we = WE_COEFFICIENT * sigma_d
        ae = compute_trajectory_length(traj)

        if not np.isfinite(we) or we == 0 or not np.isfinite(ae):
            continue

        ide = ae / we
        effective.loc[idx, "sigma_d"] = sigma_d
        effective.loc[idx, "We"] = we
        effective.loc[idx, "Ae"] = ae
        effective.loc[idx, "IDe"] = ide
        effective.loc[idx, "TPe"] = ide / mt
        effective.loc[idx, "effective_n_points"] = len(deviations)

    return effective


def to_bool_series(s):
    """
    true/false, 1/0, yes/no, success/failure等をboolに変換する．
    """
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().map({
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "success": True,
        "succeeded": True,
        "ok": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
        "failure": False,
        "failed": False,
        "error": False,
        "ng": False,
    })


def normalize_condition_from_text(x):
    """
    条件名を標準化する．
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    if s in ["1", "①", "幅広", "幅広条件", "wide", "width_wide", "w_wide"]:
        return "width_wide"
    if s in ["2", "②", "幅狭", "幅狭条件", "narrow", "width_narrow", "w_narrow"]:
        return "width_narrow"
    if s in ["3", "③", "距離大", "距離大条件", "長い", "length_long", "a_long"]:
        return "length_long"
    if s in ["4", "④", "距離小", "距離小条件", "短い", "length_short", "a_short"]:
        return "length_short"

    if "幅広" in s or "wide" in s:
        return "width_wide"
    if "幅狭" in s or "narrow" in s:
        return "width_narrow"
    if "距離大" in s or "length_long" in s or "long" in s:
        return "length_long"
    if "距離小" in s or "length_short" in s or "short" in s:
        return "length_short"

    return s


def condition_from_AWID(A, W, ID):
    """
    A, W, IDの値から条件を推定する．
    """
    if pd.isna(A) or pd.isna(W):
        return np.nan

    A = float(A)
    W = float(W)

    if np.isclose(A, 1000) and np.isclose(W, 50):
        return "width_wide"
    if np.isclose(A, 1000) and np.isclose(W, 20):
        return "width_narrow"
    if np.isclose(A, 1500) and np.isclose(W, 30):
        return "length_long"
    if np.isclose(A, 600) and np.isclose(W, 30):
        return "length_short"

    return f"A{A:g}_W{W:g}"


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
        d.groupby("condition")[params]
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
        values[cond] = arr

    if any(len(values[c]) == 0 for c in COND_ORDER):
        return pd.DataFrame()

    rows = []

    def sample_mean(cond):
        arr = values[cond]
        return rng.choice(arr, size=len(arr), replace=True).mean()

    boot_rows = []

    for _ in range(boot_n):
        bw = sample_mean("width_wide")
        bn = sample_mean("width_narrow")
        al = sample_mean("length_long")
        ast = sample_mean("length_short")

        width_diff = bw - bn
        length_diff = al - ast

        boot_rows.append({
            "width_wide_minus_width_narrow": width_diff,
            "length_long_minus_length_short": length_diff,
            "abs_width_effect_minus_abs_length_effect": abs(width_diff) - abs(length_diff),
            "same_ID20_width_wide_minus_length_short": bw - ast,
            "same_ID50_width_narrow_minus_length_long": bn - al,
            "ID20_mean_minus_ID50_mean": np.mean([bw, ast]) - np.mean([bn, al]),
        })

    boot = pd.DataFrame(boot_rows)

    for name in boot.columns:
        arr = boot[name].dropna().to_numpy()
        rows.append({
            "metric": metric,
            "param": param,
            "contrast": name,
            "mean": arr.mean(),
            "ci_low": np.percentile(arr, 2.5),
            "ci_high": np.percentile(arr, 97.5),
            "p_two_sided_approx": 2 * min(np.mean(arr <= 0), np.mean(arr >= 0)),
        })

    return pd.DataFrame(rows)


# ============================================================
# 2. CSV読み込み
# ============================================================

setup_matplotlib()

args = parse_args()
apply_cli_overrides(args)
csv_files = find_csv_files(args.csv_filenames, read_all=args.all)
if not csv_files:
    raise FileNotFoundError(f"CSV directory not found: {CSV_DIR.resolve()}")

dfs = []
for fp in csv_files:
    tmp = pd.read_csv(fp)
    tmp["source_file"] = fp.name
    dfs.append(tmp)

raw = pd.concat(dfs, ignore_index=True)
print(f"Loaded: {len(raw)} rows, {len(csv_files)} files")
print("CSV files:")
for fp in csv_files:
    print(f"  - {fp.name}")
print("Analysis settings:")
print(f"  - DEVIATION_EXCLUSION_RATIO = {DEVIATION_EXCLUSION_RATIO:g}")
print(f"  - INCLUDE_DEVIATED_TRIALS = {INCLUDE_DEVIATED_TRIALS}")
print(f"  - DELETE_OUTLIER = {DELETE_OUTLIER}")
print(f"  - OUTLIER_SIGMA = {OUTLIER_SIGMA:g}")


# ============================================================
# 3. 列名の標準化
# ============================================================

df = raw.copy()

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

mt_col = pick_col(df, [
    "mtMs", "mtSec", "MT", "mt", "movement_time", "movementTime", "duration", "time",
    "操作時間", "移動時間"
], required=True)

a_col = pick_col(df, ["amplitude", "A", "a", "distance", "path_length", "length", "経路長", "距離"], required=False)
w_col = pick_col(df, ["W", "w", "width", "path_width", "経路幅", "幅"], required=False)
id_col = pick_col(df, ["steeringId", "ID", "id", "difficulty", "difficulty_index", "D", "難易度"], required=False)

error_col = pick_col(df, [
    "errorType", "error_type", "error", "failure_type", "status", "result",
    "trial_status", "エラー", "エラー種別", "結果"
], required=False)

counted_col = pick_col(df, [
    "countedAsTrial", "is_counted", "counted", "valid_trial", "analysis_trial",
    "カウント対象", "有効試行"
], required=False)

success_col = pick_col(df, [
    "success", "is_success", "succeeded", "completed",
    "成功"
], required=False)

tpe_col = pick_col(df, ["TPe", "tpe", "effective_throughput", "throughput"], required=False)
ae_col = pick_col(df, ["Ae", "ae", "actual_path_length", "trajectory_length", "実軌跡長"], required=False)
we_col = pick_col(df, ["We", "we", "effective_width", "有効幅"], required=False)
trajectory_col = pick_col(df, ["trajectoryJson", "trajectory", "trajectory_json", "軌跡"], required=False)
deviated_col = pick_col(df, ["deviated", "is_deviated", "path_deviated", "逸脱"], required=False)


if participant_col is None:
    # 参加者IDがない場合，ファイル単位で参加者とみなす
    df["participant"] = df["source_file"]
else:
    df["participant"] = df[participant_col].astype(str)

if trial_col is None:
    df["trial_original"] = np.arange(len(df))
else:
    df["trial_original"] = pd.to_numeric(df[trial_col], errors="coerce")

df["MT"] = pd.to_numeric(df[mt_col], errors="coerce")

# MTがミリ秒らしい場合は秒に変換
if df["MT"].median(skipna=True) > 20:
    df["MT"] = df["MT"] / 1000.0

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


# ============================================================
# 4. エラー処理
# ============================================================

if error_col is not None:
    df["error_text"] = df[error_col].astype(str)
else:
    df["error_text"] = ""

if counted_col is not None:
    counted_bool = to_bool_series(df[counted_col])
    df["is_counted_trial"] = counted_bool.fillna(True)
else:
    # 操作ミス系を除外
    df["is_counted_trial"] = ~df["error_text"].apply(
        lambda x: contains_keyword(x, EXCLUDE_ERROR_KEYWORDS)
    )

if error_col is not None:
    error_is_counted_keyword = df["error_text"].apply(
        lambda x: contains_keyword(x, COUNTED_ERROR_KEYWORDS)
    )
    error_is_excluded_keyword = df["error_text"].apply(
        lambda x: contains_keyword(x, EXCLUDE_ERROR_KEYWORDS)
    )
    error_is_no_error = df["error_text"].apply(is_no_error_text)
    if COUNT_UNLISTED_COUNTED_ERRORS_AS_ERRORS:
        error_is_unlisted_counted = (
            df["is_counted_trial"]
            & ~error_is_excluded_keyword
            & ~error_is_no_error
        )
    else:
        error_is_unlisted_counted = pd.Series(False, index=df.index)

    if success_col is not None:
        success_bool = to_bool_series(df[success_col]).fillna(False)
        error_by_success = df["is_counted_trial"] & ~success_bool & ~error_is_excluded_keyword
    else:
        error_by_success = False

    df["error_counted"] = (
        error_is_counted_keyword
        | error_is_unlisted_counted
        | error_by_success
    )
    df.loc[~df["is_counted_trial"], "error_counted"] = False
elif success_col is not None:
    success_bool = to_bool_series(df[success_col]).fillna(False)
    df["error_counted"] = ~success_bool
else:
    # エラー列がない場合は，全て成功として扱う
    warnings.warn("Error/status column not found; all trials are treated as successful.")
    df["error_counted"] = False

# 試行としてカウントしないものを除外
df = df[df["is_counted_trial"]].copy()

# MTがない試行は分析不能
df = df[df["MT"].notna() & (df["MT"] > 0)].copy()

# 条件を持つ行のみ
df = df[df["condition"].notna()].copy()

# 条件順を固定
df["condition"] = pd.Categorical(df["condition"], categories=COND_ORDER, ordered=True)

# 参加者内・条件内で試行番号を振り直す
df = df.sort_values(["participant", "condition", "trial_original", "source_file"]).copy()
df["N"] = df.groupby(["participant", "condition"]).cumcount() + 1

# 400試行より後がある場合は除外
df = df[df["N"] <= MAX_TRIAL_N].copy()

if "deviationTotalMs" in df.columns:
    deviation_total_ms = pd.to_numeric(df["deviationTotalMs"], errors="coerce")
else:
    deviation_total_ms = pd.Series(np.nan, index=df.index)
mt_ms_for_deviation = df["MT"] * 1000
df["deviation_time_ratio"] = deviation_total_ms / mt_ms_for_deviation
df["exclude_by_deviation_ratio"] = (
    df["deviation_time_ratio"].notna()
    & (df["deviation_time_ratio"] >= DEVIATION_EXCLUSION_RATIO)
)

df["performance_trial_before_outlier"] = build_performance_trial_mask(
    df,
    success_col=success_col,
    deviated_col=deviated_col,
)
df["is_mt_outlier"] = mark_mt_outliers(df, df["performance_trial_before_outlier"])
df["success_for_mt"] = df["performance_trial_before_outlier"] & ~df["is_mt_outlier"]

if deviated_col is not None:
    deviated_for_tpe = to_bool_series(df[deviated_col]).fillna(False)
else:
    deviated_for_tpe = pd.Series(False, index=df.index)

if success_col is not None:
    success_bool_for_tpe = to_bool_series(df[success_col]).fillna(False)
else:
    success_bool_for_tpe = ~df["error_counted"]

# 主分析: MT分析と同じ基準を使う．
# 経路はみ出しは許容し，exclude_by_deviation_ratioのみ除外する．
df["success_clean_for_tpe"] = df["success_for_mt"]

# 補助分析: 成功していれば軽微な逸脱も含む
df["success_with_deviation_for_tpe"] = (
    success_bool_for_tpe
    & df["MT"].notna()
    & (df["MT"] > 0)
    & ~df["exclude_by_deviation_ratio"]
)

# ブロック
df["block20"] = ((df["N"] - 1) // BIN_SIZE) + 1
df["rest_block100"] = ((df["N"] - 1) // 100) + 1
df["logN"] = np.log(df["N"])

# 添付コードに合わせて，TPe = IDe / MT, IDe = Ae / We として計算する
if ae_col is not None and we_col is not None:
    df["Ae"] = pd.to_numeric(df[ae_col], errors="coerce")
    df["We"] = pd.to_numeric(df[we_col], errors="coerce")
    df["IDe"] = df["Ae"] / df["We"]
    df["TPe_clean"] = np.where(df["success_clean_for_tpe"], df["IDe"] / df["MT"], np.nan)
    df["TPe_with_deviation"] = np.where(df["success_with_deviation_for_tpe"], df["IDe"] / df["MT"], np.nan)
    df["TPe"] = df["TPe_clean"]
elif trajectory_col is not None:
    effective_clean = compute_effective_tpe_from_trajectory(
        df,
        trajectory_col,
        mask_col="success_clean_for_tpe",
    )
    effective_with_deviation = compute_effective_tpe_from_trajectory(
        df,
        trajectory_col,
        mask_col="success_with_deviation_for_tpe",
    )

    for col in ["sigma_d", "We", "Ae", "IDe", "TPe", "effective_n_points"]:
        df[col] = effective_clean[col]

    df["TPe_clean"] = effective_clean["TPe"]
    df["TPe_with_deviation"] = effective_with_deviation["TPe"]
elif tpe_col is not None:
    warnings.warn("Ae/We/trajectory not found; using the TPe column from the CSV.")
    imported_tpe = pd.to_numeric(df[tpe_col], errors="coerce")
    df["TPe_clean"] = np.where(df["success_clean_for_tpe"], imported_tpe, np.nan)
    df["TPe_with_deviation"] = np.where(df["success_with_deviation_for_tpe"], imported_tpe, np.nan)
    df["TPe"] = df["TPe_clean"]
else:
    df["TPe"] = np.nan
    df["TPe_clean"] = np.nan
    df["TPe_with_deviation"] = np.nan

df["inv_TPe"] = 1 / df["TPe"]
df["inv_TPe_with_deviation"] = 1 / df["TPe_with_deviation"]

df.to_csv(OUT_DIR / "preprocessed_trials.csv", index=False, encoding="utf-8-sig")
trial_handling_summary = pd.DataFrame([{
    "USE_SUCCESS_TRIALS_ONLY": USE_SUCCESS_TRIALS_ONLY,
    "INCLUDE_DEVIATED_TRIALS": INCLUDE_DEVIATED_TRIALS,
    "DEVIATION_EXCLUSION_RATIO": DEVIATION_EXCLUSION_RATIO,
    "DELETE_OUTLIER": DELETE_OUTLIER,
    "OUTLIER_SIGMA": OUTLIER_SIGMA,
    "n_counted_trials": len(df),
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
trial_handling_summary.to_csv(
    OUT_DIR / "trial_handling_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
print(f"Preprocessed data: {len(df)} rows")
print(df.groupby("condition")["participant"].nunique())
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


# ============================================================
# 5. 記述統計
# ============================================================

desc_trial = (
    df.groupby(["condition", "rest_block100"])
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
    .groupby(["condition", "rest_block100"])
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

# N50/N80/N90/N95
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

summary_df = pd.DataFrame(summary_rows)
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

model_texts = []

# 9-1. MT: 成功試行のみ，log(MT) ~ logN * condition
mt_model_df = df[df["success_for_mt"]].copy()
mt_model_df = mt_model_df[mt_model_df["MT"] > 0].copy()
mt_model_df["logMT"] = np.log(mt_model_df["MT"])

try:
    md = smf.mixedlm(
        "logMT ~ logN * C(condition)",
        data=mt_model_df,
        groups=mt_model_df["participant"],
        re_formula="~logN",
    )
    mdf = md.fit(method="lbfgs", maxiter=2000)
    model_texts.append("===== MixedLM: logMT ~ logN * condition =====")
    model_texts.append(str(mdf.summary()))
except Exception as e:
    model_texts.append("===== MixedLM random slope failed =====")
    model_texts.append(str(e))
    try:
        md = smf.mixedlm(
            "logMT ~ logN * C(condition)",
            data=mt_model_df,
            groups=mt_model_df["participant"],
        )
        mdf = md.fit(method="lbfgs", maxiter=2000)
        model_texts.append("===== MixedLM random intercept only =====")
        model_texts.append(str(mdf.summary()))
    except Exception as e2:
        model_texts.append("MixedLM failed completely:")
        model_texts.append(str(e2))

# 9-2. ER: GEE binomial
gee_df = df.copy()
gee_df["error_int"] = gee_df["error_counted"].astype(int)

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

# 9-3. 同一ID比較
same_id20 = mt_model_df[mt_model_df["condition"].isin(["width_wide", "length_short"])].copy()
same_id50 = mt_model_df[mt_model_df["condition"].isin(["width_narrow", "length_long"])].copy()

for label, sub in [("ID20", same_id20), ("ID50", same_id50)]:
    if len(sub) > 0 and sub["condition"].nunique() == 2:
        try:
            md = smf.mixedlm(
                "logMT ~ logN * C(condition)",
                data=sub,
                groups=sub["participant"],
            )
            mdf = md.fit(method="lbfgs", maxiter=2000)
            model_texts.append(f"\n===== Same ID comparison: {label} =====")
            model_texts.append(str(mdf.summary()))
        except Exception as e:
            model_texts.append(f"Same ID model failed: {label}")
            model_texts.append(str(e))

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

    plt.xlabel("Trial number N")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200)
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
        "length_short": "tab:red",
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
        ax.set_xlabel("Within-condition trial number N")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.suptitle(title)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
        fig.subplots_adjust(bottom=0.26, top=0.84)
    else:
        fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=220)
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
        ("MT_mean", "MT_se", "Mean MT [s]", "MT 100-trial average"),
        ("TPe_mean", "TPe_se", "Mean TPe [1/s]", "TPe 100-trial average"),
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
        ax.set_xlabel("Within-condition trial number N")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.suptitle("MT and TPe by 100-trial bins")
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
        fig.subplots_adjust(bottom=0.25, top=0.84)
    fig.savefig(OUT_DIR / "01_MT_TPe_100trial_average.png", dpi=220)
    plt.close(fig)


def plot_combined_metrics_by_bin_size(metric_bin_df):
    if metric_bin_df.empty:
        return

    metric_bin_df.to_csv(OUT_DIR / "03_metrics_bin_summary.csv", index=False, encoding="utf-8-sig")

    specs = [
        ("MT_mean", "MT_se", "Mean MT [s]", "MT"),
        ("TPe_mean", "TPe_se", "Mean TPe [1/s]", "TPe"),
        ("ER_mean", "ER_se", "Error rate [%]", "ER"),
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
            ax.set_xlabel("Within-condition trial number N")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)

        fig.suptitle(f"MT, TPe, and ER ({bin_size}-trial bins)")
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
            fig.subplots_adjust(bottom=0.25, top=0.84)
        fig.savefig(OUT_DIR / f"06_combined_metrics_bin{bin_size}.png", dpi=220)
        plt.close(fig)


def plot_same_id_hypothesis_dashboard(metric_bin_df, bin_size=HYPOTHESIS_BIN_SIZE):
    tmp = metric_bin_df[metric_bin_df["binSize"] == bin_size].copy()
    if tmp.empty:
        return

    same_id_pairs = [
        ("ID20: Wide-width A1000/W50 vs Short-distance A600/W30", ["width_wide", "length_short"]),
        ("ID50: Narrow-width A1000/W20 vs Long-distance A1500/W30", ["width_narrow", "length_long"]),
    ]
    specs = [
        ("MT_mean", "MT_se", "Mean MT [s]", "MT (lower is faster)"),
        ("TPe_mean", "TPe_se", "Mean TPe [1/s]", "TPe (higher is better)"),
        ("ER_mean", "ER_se", "Error rate [%]", "ER (lower is more stable)"),
    ]

    fig, axes = plt.subplots(len(same_id_pairs), len(specs), figsize=(17, 8.5), sharex=True)

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
            ax.set_xlabel("Within-condition trial number N")
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

    fig.suptitle(f"Same-ID hypothesis dashboard ({bin_size}-trial bins)")
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.94))
    fig.savefig(OUT_DIR / "07_same_ID_hypothesis_dashboard.png", dpi=230)
    plt.close(fig)


def plot_hypothesis_contrasts(metric_bin_df, bin_size=HYPOTHESIS_BIN_SIZE):
    tmp = metric_bin_df[metric_bin_df["binSize"] == bin_size].copy()
    if tmp.empty:
        return

    contrast_defs = [
        ("Same ID20: wide-width - short-distance", "width_wide", "length_short"),
        ("Same ID50: narrow-width - long-distance", "width_narrow", "length_long"),
        ("Width manipulation: wide - narrow", "width_wide", "width_narrow"),
        ("Distance manipulation: long - short", "length_long", "length_short"),
    ]
    metric_defs = [
        ("MT_mean", "MT difference [s]", "MT diff (negative means left is faster)"),
        ("TPe_mean", "TPe difference [1/s]", "TPe diff (positive means left is better)"),
        ("ER_mean", "ER difference [%]", "ER diff (negative means left is more stable)"),
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
        ax.set_xlabel("Within-condition trial number N")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Hypothesis contrasts over trials ({bin_size}-trial bins)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "08_hypothesis_contrasts_over_trials.png", dpi=230)
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
            "ID_group": "ID20" if cond in ["width_wide", "length_short"] else "ID50",
            "MT_mean": row.get("MT_mean", np.nan),
            "TPe_mean": row.get("TPe_mean", np.nan),
            "ER_mean": row.get("ER_mean", np.nan),
        })

    final_df = pd.DataFrame(final_rows)
    if final_df.empty:
        return
    final_df.to_csv(OUT_DIR / "09_hypothesis_final_100trial_summary.csv", index=False, encoding="utf-8-sig")

    specs = [
        ("MT_mean", "Mean MT [s]", "Final 100 trials: MT"),
        ("TPe_mean", "Mean TPe [1/s]", "Final 100 trials: TPe"),
        ("ER_mean", "Error rate [%]", "Final 100 trials: ER"),
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

    fig.suptitle("Hypothesis check: final 100-trial summary")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "09_hypothesis_final_100trial_summary.png", dpi=230)
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
                "MT [s]",
                cond_data[cond_data["success_for_mt"] & cond_data["MT"].notna()][["N", "MT"]],
                "MT",
                "N",
                10,
            ),
            (
                "TPe",
                "TPe [1/s]",
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
                "ER [%]",
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
                "MT [s]",
                cond_data[cond_data["success_for_mt"] & cond_data["MT"].notna()],
                "MT",
                cond_100,
                "MT_mean",
                "MT_se",
            ),
            (
                "TPe",
                "TPe [1/s]",
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
                "ER [%]",
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

        axes[-1].set_xlabel("Within-condition trial number N")
        fig.suptitle(f"{condition_number_label(cond)}: single trials, 100-trial average, nonlinear fit")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(
            OUT_DIR / f"10_condition_{COND_NUMBER.get(cond)}_MT_TPe_ER_1trial_100trial_fit.png",
            dpi=230,
        )
        plt.close(fig)


SINGLE_TRIAL_METRIC_SPECS = [
    ("MT", "MT_single", "MT [s]", "MT per trial"),
    ("TPe", "TPe_single", "TPe [1/s]", "TPe per trial"),
    ("ER", "ER_single", "ER [%]", "ER per trial"),
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

        axes[-1].set_xlabel("Within-condition trial number N")
        fig.suptitle(f"{condition_number_label(cond)}: MT, TPe, and ER per trial with linear regression")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(
            OUT_DIR / f"12_condition_{COND_NUMBER.get(cond)}_MT_TPe_ER_single_trial.png",
            dpi=230,
        )
        plt.close(fig)

    write_single_trial_outputs(single_frames, fit_rows)


def plot_same_id_mt_tpe_evaluation(data, metric_bin_df, fit_map, fit_condition_df):
    pairs = [
        ("ID20", ["width_wide", "length_short"]),
        ("ID50", ["width_narrow", "length_long"]),
    ]
    metrics = [
        ("MT", "MT [s]", "MT", "MT_mean", "MT_se", "lower is faster"),
        ("TPe", "TPe [1/s]", "TPe_clean", "TPe_mean", "TPe_se", "higher is better"),
    ]

    fig, axes = plt.subplots(len(pairs), len(metrics), figsize=(15, 8.5), sharex=True)

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
            ax.set_xlabel("Within-condition trial number N")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=7)

    fig.suptitle("Same-ID MT and TPe evaluation (single trials + 100-trial average + nonlinear fit)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "11_same_ID_MT_TPe_evaluation_1trial_100trial_fit.png", dpi=230)
    plt.close(fig)

    same_id_fit = fit_condition_df[
        fit_condition_df["metric"].isin(["MT", "TPe"])
        & fit_condition_df["ID_group"].isin(["ID20", "ID50"])
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
        "ylabel": "MT [s]",
        "title": "MT",
        "filename": "02_MT",
    },
    "TPe": {
        "value_col": "TPe_clean",
        "mask_col": "success_clean_for_tpe",
        "ylabel": "TPe [1/s]",
        "title": "TPe",
        "filename": "04_TPe",
    },
    "We": {
        "value_col": "We",
        "mask_col": None,
        "ylabel": "We [px]",
        "title": "We",
        "filename": "03_We",
    },
    "Ae": {
        "value_col": "Ae",
        "mask_col": None,
        "ylabel": "Ae [px]",
        "title": "Ae",
        "filename": "05_Ae",
    },
}


def add_continuous_trial_index(data):
    sort_cols = [
        col for col in [
            "source_file",
            "participant",
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
    success_bool = to_bool_series(data["success"]).fillna(False) if "success" in data.columns else ~data["error_counted"]
    deviated_bool = to_bool_series(data["deviated"]).fillna(False) if "deviated" in data.columns else pd.Series(False, index=data.index)

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
            "n_trials": len(sub),
            "n_success": int(success_sub.sum()),
            "success_rate_percent": float(success_sub.mean() * 100) if len(sub) else np.nan,
            "n_error_counted": int(sub["error_counted"].sum()),
            "error_rate_percent": float(sub["error_counted"].mean() * 100) if len(sub) else np.nan,
            "n_deviated": int(deviated_sub.sum()),
            "deviation_rate_percent": float(deviated_sub.mean() * 100) if len(sub) else np.nan,
            "deviation_count_mean": numeric_or_nan(sub, "deviationCount").mean(),
            "deviation_count_sum": numeric_or_nan(sub, "deviationCount").sum(),
            "deviation_time_ms_mean": numeric_or_nan(sub, "deviationTotalMs").mean(),
            "deviation_time_ms_sum": numeric_or_nan(sub, "deviationTotalMs").sum(),
            "deviation_time_ratio_mean": numeric_or_nan(sub, "deviation_time_ratio").mean(),
            "n_excluded_by_deviation_ratio": int(sub["exclude_by_deviation_ratio"].sum()),
            "deviation_ratio_exclusion_threshold": DEVIATION_EXCLUSION_RATIO,
            "max_deviation_px_mean": numeric_or_nan(sub, "maxDeviationPx").mean(),
            "max_deviation_px_max": numeric_or_nan(sub, "maxDeviationPx").max(),
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
            "n_deviation_error_text": int(sub["error_text"].eq("deviation").sum()),
            "n_excessive_deviation_error_text": int(sub["error_text"].eq("excessive_deviation").sum()),
            "n_deviated_true": int(to_bool_series(sub["deviated"]).fillna(False).sum()) if "deviated" in sub.columns else np.nan,
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
        ("success_rate_percent", "Success rate [%]"),
        ("error_rate_percent", "Error rate [%]"),
        ("deviation_rate_percent", "Deviation rate [%]"),
    ]
    for ax, (col, title) in zip(axes, rate_specs):
        ax.bar(x, summary[col], color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Condition summary: success, error, and deviation rates")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(CONDITION_REPORT_DIR / "01_condition_rates.png", dpi=230)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    mean_specs = [
        ("mt_mean", "Mean MT [s]"),
        ("tpe_mean", "Mean TPe [1/s]"),
    ]
    for ax, (col, title) in zip(axes, mean_specs):
        ax.bar(x, summary[col], color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Condition summary: mean MT and TPe")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(CONDITION_REPORT_DIR / "01_condition_mean_MT_TPe.png", dpi=230)
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
        ax.set_xlabel("Within-condition trial number N")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(CONDITION_REPORT_DIR / f"{spec['filename']}_condition_{COND_NUMBER.get(cond)}_trend.png", dpi=230)
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
    ax.set_xlabel("Within-condition trial number N")
    ax.set_ylabel(spec["ylabel"])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(CONDITION_REPORT_DIR / f"{spec['filename']}_all_conditions_trend.png", dpi=230)
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
        fig.savefig(CONDITION_REPORT_DIR / f"{spec['filename']}_mean_by_condition.png", dpi=230)
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
    ax.set_xlabel("Continuous trial number")
    ax.set_ylabel(spec["ylabel"])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(CONDITION_REPORT_DIR / f"{spec['filename']}_continuous_all_trials.png", dpi=230)
    plt.close(fig)

    return fit_rows


def minmax_normalize(series):
    series = pd.to_numeric(series, errors="coerce")
    valid = series.dropna()
    if valid.empty or valid.max() == valid.min():
        return pd.Series(np.nan, index=series.index)
    return (series - valid.min()) / (valid.max() - valid.min())


def plot_we_ae_tpe_relationships(report_data):
    block = (
        report_data.groupby("global_block100", observed=True)
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
        (axes[0], "We_mean", "Mean We [px]", "We by 100-trial blocks"),
        (axes[1], "Ae_mean", "Mean Ae [px]", "Ae by 100-trial blocks"),
    ]:
        ax.plot(block["global_N"], block[ycol], marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Continuous trial number")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(CONDITION_REPORT_DIR / "05_We_Ae_100trial_blocks.png", dpi=230)
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
    ax.set_title("Normalized MT, We, Ae, and TPe over all trials")
    ax.set_xlabel("Continuous trial number")
    ax.set_ylabel("Min-max normalized value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(CONDITION_REPORT_DIR / "05_normalized_MT_We_Ae_TPe.png", dpi=230)
    plt.close(fig)

    scatter_data = report_data[report_data["TPe_clean"].notna()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, xcol, xlabel, title in [
        (axes[0], "We", "We [px]", "TPe vs We"),
        (axes[1], "Ae", "Ae [px]", "TPe vs Ae"),
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
        ax.set_ylabel("TPe [1/s]")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(CONDITION_REPORT_DIR / "05_TPe_vs_We_Ae_scatter.png", dpi=230)
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
    ylabel="MT successful trials only [s]",
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
    ylabel="ER",
    filename="learning_curve_ER.png"
)

# 1/TPe main analysis
if df["inv_TPe"].notna().sum() > 0:
    plot_block_metric(
        df[df["success_clean_for_tpe"] & df["inv_TPe"].notna()],
        ycol="inv_TPe",
        ylabel="1 / TPe main analysis",
        filename="learning_curve_inv_TPe.png"
    )

# 1/TPe auxiliary analysis
if df["inv_TPe_with_deviation"].notna().sum() > 0:
    plot_block_metric(
        df[df["success_with_deviation_for_tpe"] & df["inv_TPe_with_deviation"].notna()],
        ycol="inv_TPe_with_deviation",
        ylabel="1 / TPe auxiliary analysis including deviated successes",
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
        ylabel="Mean MT [s]",
        title="MT averaged by 10/50/100-trial bins",
        filename="03_MT_10_50_100_average.png",
    )
    plot_metric_bin_size_lines(
        metric_bin_df,
        ycol="TPe_mean",
        secol="TPe_se",
        ylabel="Mean TPe [1/s]",
        title="TPe averaged by 10/50/100-trial bins (clean successes)",
        filename="04_TPe_10_50_100_average.png",
    )
    plot_metric_bin_size_lines(
        metric_bin_df,
        ycol="ER_mean",
        secol="ER_se",
        ylabel="Error rate [%]",
        title="Error rate averaged by 10/50/100-trial bins",
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
    id_group = "ID20" if cond in ["width_wide", "length_short"] else "ID50"

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
