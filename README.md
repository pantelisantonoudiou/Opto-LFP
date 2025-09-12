# Opto‑LFP Analysis Pipeline

Build a **multi‑folder index** of `.adicht` recordings, compute **stim/baseline power ratios**, then make **seaborn plots**. Everything runs from a Conda prompt, with the last plotting step in Spyder.

---

## TL;DR — Run the pipeline

1. **Build the unified index** (GUI + per‑folder scan)

   ```bash
   conda activate optolfp # there are instructions below on how to create it, alternatively install the toolboxes in your base environment
   python multi_folder_index.py
   ```

   • Pick your `root_dir` in the GUI, confirm the detected folders.
   • Output is saved in `root_dir` (default script uses `unified_index.csv`; you can rename to `index.csv`).

2. **Compute power ratios** (CLI prompts)

   ```bash
   python power_ratio_from_index.py
   # prompts
   # root_dir = <same root_dir>
   # index_name [index.csv]         # enter your index filename (e.g., unified_index.csv)
   # output_name [power_ratio_results.csv]
   ```

   • Produces `power_ratio_results.csv` with **all original index columns** plus: `stim_power`, `baseline_power`, `power_ratio`.

3. **Plot in Spyder** (no saving by default)

   * Open Spyder → run `power_ratio_plots.py` → enter `root_dir` and results filename when prompted.
   * You’ll get two figures (faceting by brain region and by **condition** using `col`/`col_wrap`).

---

## Data requirements & organization

**Per‑file assumptions**

* One `.adicht` file represents one animal (recommended).
* Each file contains **multiple blocks** possible.
* Each file must contain **exactly one** stim channel that matches your GUI input substring (case‑insensitive).
* Data channels are selected by **include keywords**; unwanted channels are removed by **drop keywords**.

**Folder layout**

```
root_dir/
  ├─ <condition‑A>/
  │    ├─ animal_001.adicht
  │    └─ animal_002.adicht
  ├─ <condition‑B>/
  │    └─ animal_003.adicht
  ├─ unified_index.csv   # or index.csv (produced by step 1)
  └─ power_ratio_results.csv  # produced by step 2
```

* If `root_dir` has **no** subfolders, it is treated as a single **condition**.

**Index content (minimum columns expected by step 2)**

* `filename` (no extension), `condition`, `block`, `channel_id`, `start_time`, `stop_time`, `stim_hz`

**Results content (step 2 output)**

* **All index columns**, plus: `stim_power`, `baseline_power`, `power_ratio`

---

## Scripts in this repo

### `index_inputs_gui.py`

CustomTkinter GUI to collect:

* `root_dir` (data directory)
* `channel_names` (comma‑separated include keywords; **data channels only**)
* `stim_channel_name` (single substring, e.g., "opto" or "stim")
* `drop_keywords` (comma‑separated substrings to exclude channels like `empty, bio, drop`)

Includes a safe shutdown to avoid Tk “after script” warnings.

### `multi_folder_index.py`

* Uses the GUI to get settings.
* Scans **immediate subfolders** of `root_dir` as conditions (uses `root_dir` itself if no subfolders).
* Validates exactly **one** stim channel per file; builds a per‑file/per‑block channel index; drops unwanted channels.
* Parses stim trains and merges per‑block stim metadata.
* Saves the unified index to `root_dir` (default shown earlier as `unified_index.csv`).

### `power_ratio_from_index.py`

* Prompts for `root_dir`, `index_name` (default `index.csv`), `output_name` (default `power_ratio_results.csv`).
* For each index row, loads the data from `root_dir/condition/filename.adicht`.
* Computes STFT band‑power around `stim_hz` during **stim** and **baseline** windows, then calculates `power_ratio = stim_power / baseline_power`.
* Writes `power_ratio_results.csv` with **all original index columns** plus the computed fields.
* STFT parameters are fixed in the script (e.g., window 0.5 s, overlap 0.5, ±1 Hz band around `stim_hz`, 2 s baseline).

### `power_ratio_plots.py`

* Prompts for `root_dir` and results filename (default `power_ratio_results.csv`).
* If `animal_id` is missing but `filename` exists, it uses `filename` as `animal_id`.
* Makes two seaborn `relplot` figures:

  1. `stim_hz` vs `power_ratio`, error bars = SE; faceted by `brain_region` (if present), hue by `animal_id` (if present).
  2. Same metric, **faceted by `condition`** via `col` with `col_wrap`.
* Shows figures (does **not** save by default).

### (Optional) `build_index.py`, `analyze_opto_lfp.py`

* `build_index.py` contains the underlying parsing utilities (`IndexMaker`, `parse_stims`, `build_stim_index`).

---

## Environment & toolboxes

Create a new Conda environment (recommended):

```bash
conda create -n optolfp python=3.10 -y
conda activate optolfp
```

Install dependencies:

```bash
# Core scientific stack
conda install -y numpy pandas scipy matplotlib seaborn tqdm

# ADI file reader + GUI
pip install adi-reader customtkinter

# Optional (for running the final plots step in an IDE):
conda install -y spyder
```

---

## Detailed run instructions

### 1) Build the index

```bash
python multi_folder_index.py
```

* In the GUI:

  * **Data directory** → choose `root_dir` (the parent containing condition subfolders or `.adicht` files)
  * **Channels** → comma‑separated include terms (case‑insensitive)
  * **Stim channel** → a single substring (e.g., `opto`)
  * **Keywords to drop** → terms like `empty, bio, drop`
* Confirm the list of detected condition folders when prompted.
* Check `root_dir` for the resulting index CSV (e.g., `index.csv`).

### 2) Compute power ratios

```bash
python power_ratio_from_index.py
# root_dir = <your root_dir>
# index_name [index.csv]  # enter your actual index filename if different
# output_name [power_ratio_results.csv]
```

### 3) Plot (Spyder)

* Open Spyder → run `power_ratio_plots.py` → answer prompts for `root_dir` and results filename.
* Inspect figures; use Spyder’s saving tools if you want to export.

---

## Troubleshooting

* **No stim channel found / multiple stim channels**: adjust `stim_channel_name` or data; the index builder expects exactly one stim channel per file.
* **No rows in results**: ensure index columns include at least `filename, condition, block, channel_id, start_time, stop_time, stim_hz` and that files are placed under `root_dir/condition/filename.adicht`.
* **Different index filename**: at the step‑2 prompt, enter your exact index filename (e.g., `unified_index.csv`).

---

## Repro checklist

* [ ] `root_dir` laid out as above; `.adicht` files under condition subfolders
* [ ] Built index via `multi_folder_index.py` (has required columns)
* [ ] Ran `power_ratio_from_index.py` to produce `power_ratio_results.csv`
* [ ] Opened `power_ratio_plots.py` in Spyder and ran the figures
