# power_ratio_from_index.py
# -*- coding: utf-8 -*-
"""
Compute per-trial power ratio (stim / baseline) using a unified index.

Data / Index Contract
---------------------
The unified index CSV (e.g., 'index.csv') lives in the **parent directory** (root_dir),
and contains at least the following columns (case-sensitive):
- filename       : str, base filename without extension (e.g., 'mouse01_day1')
- condition      : str, subfolder name under root_dir where the file resides
- block          : int, ADI block number (0-based here; ADI API uses 1-based for get_data)
- channel_id     : int, channel index within the ADI file
- start_time     : int, sample index where the stim window begins (inclusive)
- stop_time      : int, sample index where the stim window ends   (exclusive)
- stim_hz        : int or float, nominal stimulation frequency in Hz

Folder Organization
-------------------
root_dir/
  index.csv                        # the index this script reads
  <condition-1>/
    <filename-1>.adicht
    <filename-2>.adicht
    ...
  <condition-2>/
    ...
  ...

Output
------
Writes 'power_ratio_results.csv' in root_dir with:
- **all original columns from the index**, plus:
- stim_power, baseline_power, power_ratio
"""

# ================================ imports ================================ #
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import adi
from scipy.signal import stft
# ======================================================================== #

# ------------------------- fixed processing params ----------------------- #
# (Leave these fixed as requested)
win_s = 0.5             # STFT window (sec)
overlap = 0.5           # STFT overlap fraction [0..1)
band_margin = 1.0       # +/- Hz around stim_hz for band power
baseline_s = 2.0        # seconds before start_time for baseline window
outlier_pct = (0.18, 99.9)  # percentile cutoffs for interpolation
# ------------------------------------------------------------------------ #


# ---------------------------- core helpers ---------------------------- #
def get_data(path: str, data_ch: int, start: int, stop: int, block: int):
    """Load samples from a channel/block in a LabChart .adicht file."""
    fread = adi.read_file(path)
    ch = fread.channels[data_ch]
    fs = ch.fs[block]
    data = ch.get_data(block + 1, start_sample=int(start), stop_sample=int(stop))
    return data, fs


def remove_outliers_linear(data: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    """Replace outliers outside [p_lo, p_hi] percentiles via linear interpolation."""
    data = data.astype(float, copy=True)
    lo = np.percentile(data, p_lo)
    hi = np.percentile(data, p_hi)
    mask = (data < lo) | (data > hi)
    if mask.any():
        data[mask] = np.nan
        data = pd.Series(data).interpolate(method="linear", limit_direction="both").values
    return data


def band_power_stft(data: np.ndarray, fs: float, center_hz: float, margin_hz: float,
                    win_s: float, overlap: float) -> float:
    """Mean power in a narrow band around center_hz using STFT."""
    nperseg = max(8, int(fs * win_s))
    noverlap = int(nperseg * overlap)
    f, t, z = stft(data, fs, nperseg=nperseg, noverlap=noverlap)
    p = np.abs(z) ** 2
    f_lo, f_hi = center_hz - margin_hz, center_hz + margin_hz
    sel = (f >= f_lo) & (f <= f_hi)
    if not np.any(sel):
        return float("nan")
    return float(np.nanmean(p[sel, :]))


# ------------------------ main computation logic ----------------------- #
def compute_power_ratio_df(index_df: pd.DataFrame, root_dir: str) -> pd.DataFrame:
    """
    Compute stim/baseline power ratio for each row in index_df and return a DataFrame.
    The returned DataFrame contains **all original index columns** plus:
    ['stim_power', 'baseline_power', 'power_ratio'].

    Assumes files are located at root_dir / condition / (filename + '.adicht').
    """
    required = {"filename", "condition", "block", "channel_id", "start_time", "stop_time", "stim_hz"}
    missing = sorted(c for c in required if c not in index_df.columns)
    if missing:
        raise ValueError(f"Index missing required columns: {missing}")

    rows = []
    for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="computing power ratios"):
        # Start from the full index row, then append computed fields
        rec = row.to_dict()

        file_path = os.path.join(root_dir, str(row["condition"]), f"{row['filename']}.adicht")

        # --- stim segment ---
        stim_data, fs = get_data(
            file_path,
            int(row["channel_id"]),
            int(row["start_time"]),
            int(row["stop_time"]),
            int(row["block"])
        )
        stim_data = remove_outliers_linear(stim_data, *outlier_pct)
        stim_power = band_power_stft(stim_data, fs, float(row["stim_hz"]), band_margin, win_s, overlap)

        # --- baseline segment (immediately before start_time) ---
        base_stop = int(row["start_time"])
        base_start = max(0, base_stop - int(baseline_s * fs))
        base_data, _ = get_data(file_path, int(row["channel_id"]), base_start, base_stop, int(row["block"]))
        base_data = remove_outliers_linear(base_data, *outlier_pct)
        base_power = band_power_stft(base_data, fs, float(row["stim_hz"]), band_margin, win_s, overlap)

        pr = float("nan")
        if base_power not in (0.0, None) and np.isfinite(base_power) and base_power != 0.0:
            pr = float(stim_power / base_power)

        # Append computed fields
        rec["stim_power"] = float(stim_power)
        rec["baseline_power"] = float(base_power)
        rec["power_ratio"] = pr

        rows.append(rec)

    out_df = pd.DataFrame(rows)

    # Column order: original index columns first, then computed fields
    computed_cols = ["stim_power", "baseline_power", "power_ratio"]
    ordered_cols = list(index_df.columns) + [c for c in computed_cols if c not in index_df.columns]
    out_df = out_df.reindex(columns=ordered_cols)

    return out_df


# --------------------------------- main -------------------------------- #
if __name__ == "__main__":
    # take the 3 inputs with **lowercase** names; keep STFT params fixed
    def _ask(prompt, default=None, required=False):
        s = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip().strip("\"'")
        if not s:
            if required and default is None:
                raise ValueError(f"{prompt} is required.")
            return default
        return s

    root_dir = _ask("root_dir", required=True)
    index_name = _ask("index_name", default="index.csv")
    output_name = _ask("output_name", default="power_ratio_results.csv")

    inputs = {"root_dir": root_dir, "index_name": index_name, "output_name": output_name}

    index_path = os.path.join(inputs["root_dir"], inputs["index_name"])
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")

    index_df = pd.read_csv(index_path)
    out_df = compute_power_ratio_df(index_df, inputs["root_dir"])

    out_path = os.path.join(inputs["root_dir"], inputs["output_name"])
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved power ratio results to:\n  {out_path}")
    print(f"Shape: {out_df.shape[0]} rows x {out_df.shape[1]} columns")
