# multi_folder_index.py
# -*- coding: utf-8 -*-
"""
Build a unified stimulation index across multiple subfolders.

Data/File Contract
------------------
- Each `.adicht` file corresponds to one animal.
- Each file may contain multiple blocks (recordings).
- Each file must contain exactly one stimulation channel (e.g., opto TTL).
- Files may include unwanted channels; these are removed via `drop_keywords`.

Folder Contract
---------------
- `root_dir` may contain multiple immediate subfolders; each subfolder is a separate 'condition'.
- If no subfolders exist, `root_dir` itself is treated as one condition.
- The unified index is saved to `root_dir` (default: `unified_index.csv`).
"""

# =============================================================================
#                                 Imports
# =============================================================================
import os
import pandas as pd
from tqdm import tqdm
from build_index import IndexMaker, build_stim_index
from index_inputs_gui import run_index_inputs_gui
# =============================================================================
# =============================================================================


def build_multi_folder_index(settings, stim_settings, *, save=True, output_name="unified_index.csv", confirm=True):
    """
    Build a unified stimulation index by iterating over immediate subfolders of `settings['root_dir']`.
    For each folder: validate stim channel(s), build an index, parse stim trains, and tag with 'condition'.

    Parameters
    ----------
    settings : dict
        Expected keys:
          - 'root_dir' (str)
          - 'channel_names' (list[str])
          - 'drop_keywords' (list[str])
          - 'stim_channel_name' (str)
    stim_settings : dict
        Expected keys:
          - 'stim_threshold' (float)
          - 'min_freq' (float)
          - 'prominence' (float)
    save : bool, default True
        If True, save the unified index to `root_dir/output_name`.
    output_name : str, default "unified_index.csv"
        Filename for the saved index.
    confirm : bool, default True
        If True, print folders and prompt the user to proceed.

    Returns
    -------
    tuple[str, pandas.DataFrame] | None
        (saved_path, unified_index) if data was built; None if aborted or no data.
    """
    root_dir = settings["root_dir"]

    # Discover immediate subdirectories (conditions); if none, use root itself
    subdirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    if not subdirs:
        subdirs = [root_dir]

    folder_names = [
        os.path.basename(p.rstrip(os.sep)) or os.path.basename(root_dir.rstrip(os.sep))
        for p in subdirs
    ]

    # Print and confirm
    if confirm:
        print("\nFolders detected as conditions:")
        for name in folder_names:
            print(f"  - {name}")
        proceed = input("\nProceed with these folders? [y/N]: ").strip().lower()
        if proceed not in {"y", "yes"}:
            print("Aborted by user.")
            return None

    per_folder_indices = []

    # Iterate folders
    for folder_path, folder_name in tqdm(list(zip(subdirs, folder_names)), desc="Processing folders"):
        try:
            idx_maker = IndexMaker(
                root_dir=folder_path,
                channel_names=settings["channel_names"],
                drop_keywords=settings["drop_keywords"],
                stim_channel_name=settings["stim_channel_name"],
            )

            # Validate exactly one stim channel per file
            idx_maker.iterate_files(idx_maker.stim_channel_check)

            # Build channel index and drop unwanted channels
            file_index = idx_maker.iterate_files(idx_maker.get_properties)
            if file_index is None or file_index.empty:
                print(f"[WARN] No properties found in folder: {folder_name}")
                continue
            filt_index = idx_maker.drop_rows(file_index)

            # Parse stim trains and merge metadata
            merged = build_stim_index(filt_index, idx_maker, stim_settings)
            if merged is None or merged.empty:
                print(f"[WARN] No stim data parsed in folder: {folder_name}")
                continue

            merged["condition"] = folder_name
            per_folder_indices.append(merged)

        except Exception as e:
            print(f"[ERROR] Skipping folder '{folder_name}' due to error: {e}")

    if not per_folder_indices:
        print("No data found across folders; unified index not created.")
        return None

    unified_index = pd.concat(per_folder_indices, axis=0).reset_index(drop=True)

    saved_path = None
    if save:
        saved_path = os.path.join(root_dir, output_name)
        unified_index.to_csv(saved_path, index=False)
        print(f"\nUnified index saved to:\n  {saved_path}")
        print(f"Shape: {unified_index.shape[0]} rows x {unified_index.shape[1]} columns")

    return (saved_path, unified_index)


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1) Collect user-defined settings from GUI:
    #    - root_dir:      base directory with `.adicht` files or subfolders
    #    - channel_names: channels/brain regions to include
    #    - drop_keywords: substrings for channels to drop
    #    - stim_channel_name: stim channel identifier (e.g. "opto")
    # ------------------------------------------------------------------
    settings = run_index_inputs_gui()
    if settings is None:
        raise Exception('No User Inputs. Aborting operation.')
    

    # ------------------------------------------------------------------
    # 2) Define stimulation detection parameters:
    #    - stim_threshold: amplitude threshold for TTL conversion
    #    - min_freq:       minimum frequency (Hz) separating stim trains
    #    - prominence:     threshold for detecting rising/falling edges
    # ------------------------------------------------------------------
    stim_settings = {
        "stim_threshold": 0.05,
        "min_freq": 1.0,
        "prominence": 1.0,
    }

    # ------------------------------------------------------------------
    # 3) Build the unified index across all subfolders:
    #    - validate stim channels
    #    - build per-file, per-block channel index
    #    - drop unwanted channels
    #    - parse stim trains
    #    - merge into a single DataFrame
    #    The function also saves the output to `root_dir/unified_index.csv`.
    # ------------------------------------------------------------------
    result = build_multi_folder_index(
        settings,
        stim_settings,
        save=True,
        output_name="index.csv",
        confirm=True
    )