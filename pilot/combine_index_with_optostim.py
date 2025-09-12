# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import sys
import adi
import numpy as np
import pandas as pd
from tqdm import tqdm
##### ------------------------------------------------------------------- #####


def parse_stims(file_path, lfp_ch, stim_ch, block, min_freq=1,
                prominence=1, stim_threshold=2, ttl_height=5):
    """
    Parse stimulation trains from a LabChart file.

    Detects stimulation trains in a specified channel, reconstructs a TTL-like
    signal, extracts train start/stop sample indices, estimates the train
    frequency, and returns a summary DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the LabChart file to be parsed.
    lfp_ch : int
        Index of the LFP channel (used to extract the animal ID).
    stim_ch : int
        Index of the stimulation channel (1-based in this implementation).
    block : int
        Block number to read from the LabChart file.
    min_freq : float, optional (default=1)
        Minimum frequency (Hz) used to separate distinct trains.
        Internally converted to a minimum gap (samples) between trains.
    prominence : float, optional (default=1)
        Threshold applied to the TTL diff to detect rising/falling edges.
    stim_threshold : float, optional (default=2)
        Amplitude threshold applied to the stim channel to create a boolean
        TTL-like signal.
    ttl_height : float, optional (default=5)
        Height assigned to the TTL signal when above threshold.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns:
        - animal_id : str
        - stim_hz   : int
        - start_time: int (sample index)
        - stop_time : int (sample index)
        - block     : int

    Notes
    -----
    - Frequency is estimated from the median inter-event interval (IEI) of
      pulses within each train.
    - animal_id is derived from the LFP channel name assuming the format
      "<prefix>-<id>-<suffix>" and will extract the middle token.
    """
    # Read LabChart file
    fread = adi.read_file(file_path)

    # Select stimulation channel (NOTE: using 1-based input => convert to 0-based)
    ch_obj = fread.channels[stim_ch - 1]
    stim = ch_obj.get_data(block + 1)  # ADI blocks are often 1-based
    fs = ch_obj.fs[block]              # sampling rate for this block

    # Threshold to build a boolean(TTL-like) stream and find transitions
    stim_bool = (stim > stim_threshold) * ttl_height
    locs = np.where(np.diff(stim_bool) > prominence)[0] + 1  # rising edges

    # Recreate TTL train envelopes by grouping pulses with a min-gap
    min_gap = int(fs / min_freq)  # samples: gap threshold derived from min_freq
    start = locs[np.append(True, np.diff(locs) > min_gap)]
    stop = locs[np.append(np.diff(locs) > min_gap, True)]
    ttl = np.zeros(stim.shape[0])
    for x, y in zip(start, stop):
        ttl[x:y] = ttl_height

    # Find TTL edges (train boundaries)
    ttl_diff = ttl[1:] - ttl[:-1]
    start = np.where(ttl_diff > prominence)[0] + 1
    stop = np.where(-ttl_diff > prominence)[0] + 1

    # Estimate per-train frequency from median IEI
    freqs = []
    for x1, x2 in zip(start, stop):
        train = locs[(locs >= x1) & (locs < x2)]
        # If <2 pulses, frequency is undefined -> fall back to 1 to avoid div/0
        if train.size < 2:
            freqs.append(1)
            continue
        iei = np.median(np.diff(train)) / fs
        freq = int(np.ceil(1 / iei)) if iei > 0 else 1
        freqs.append(freq)

    # Buffer from first peak: one period (in samples)
    time_from_first_peak = (fs / np.array(freqs)).astype(int)

    # Get animal id from LFP channel name
    animal_id_tokens = fread.channels[lfp_ch].name.split('-')
    animal_id = ('-' + animal_id_tokens[1] + '-') if len(animal_id_tokens) > 1 else fread.channels[lfp_ch].name

    # Build output dataframe
    df = pd.DataFrame({
        'animal_id': np.repeat([animal_id], len(freqs)),
        'stim_hz': freqs,
        'start_time': start - time_from_first_peak,
        'stop_time': stop + time_from_first_peak,
        'block': block
    })
    return df


def parse_multiple_files(parent_path, index_df, stim_ch, stim_threshold=1,
                         min_freq=1, prominence=1, show_progress=True):
    """
    Detect stim trains across all files listed in an index DataFrame.

    Parameters
    ----------
    parent_path : str
        Root directory containing the data folders/files.
    index_df : pandas.DataFrame
        Must include columns: folder_path, file_name, channel_id, block, animal_id.
    stim_ch : int
        Stimulation channel index (1-based, consistent with parse_stims).
    stim_threshold : float, optional
        Threshold to binarize the stim signal.
    min_freq : float, optional
        Minimum frequency (Hz) for separating trains.
    prominence : float, optional
        Edge detection threshold on TTL diffs.
    show_progress : bool, optional
        If True, wrap iteration in a tqdm progress bar.

    Returns
    -------
    pandas.DataFrame
        Concatenation of per-file detection results with an extra 'file_name' column.
        Empty DataFrame if no rows are produced.
    """
    # Choose iterator with or without progress bar
    iterator = index_df.iterrows()
    if show_progress:
        iterator = tqdm(index_df.iterrows(), total=len(index_df), desc="Parsing files")

    results = []
    for _, row in iterator:
        # Compose absolute path to each recording
        file_path = os.path.join(parent_path, row.folder_path, row.file_name)

        # Run detection for this file
        df_one = parse_stims(
            file_path=file_path,
            lfp_ch=row.channel_id,
            stim_ch=stim_ch,
            min_freq=min_freq,
            prominence=prominence,
            stim_threshold=stim_threshold,
        )
        # Keep file_name to enable later join/merge
        df_one['file_name'] = row.file_name
        results.append(df_one)

    if not results:
        return pd.DataFrame(columns=['animal_id', 'stim_hz', 'start_time', 'stop_time', 'block', 'file_name'])

    return pd.concat(results, axis=0, ignore_index=True)


def combine_with_index(index_df, detected_df):
    """
    Combine detected stim info with the original index on (animal_id, block, file_name),
    preferring detected fields where available.

    Parameters
    ----------
    index_df : pandas.DataFrame
        Original index with at least: animal_id, block, file_name.
    detected_df : pandas.DataFrame
        Output from parse_multiple_files / parse_stims including: animal_id, block, file_name, stim_hz.

    Returns
    -------
    pandas.DataFrame
        A DataFrame aligned to the original index order/rows, with an added 'stim_hz' column.
    """
    combined_rows = []
    for i, row in index_df.iterrows():
        # Match on animal_id, block, file_name
        mask = (
            (detected_df['animal_id'] == row.animal_id) &
            (detected_df['block'] == row.block) &
            (detected_df['file_name'] == row.file_name)
        )
        match = detected_df[mask]

        # Convert the index row (Series) into a single-row DataFrame
        row_df = pd.DataFrame(dict(zip(row.index, row.values)), index=[i])

        # Prefer detected data where present (join on animal_id)
        merged = match.set_index('animal_id').combine_first(row_df.set_index('animal_id'))
        combined_rows.append(merged.reset_index())

    out_df = pd.concat(combined_rows, axis=0)

    # Ensure output columns mirror the original index plus stim_hz at the end
    col_order = index_df.columns.tolist()
    if 'stim_hz' not in col_order:
        col_order = col_order + ['stim_hz']
    out_df = out_df[col_order]
    return out_df


if __name__ == '__main__':
    # ------------------------------- STIM SETTINGS ------------------------------ #
    # Edit these defaults as needed for your dataset. They are NOT prompted.
    # - stim_threshold : amplitude threshold to binarize the stim channel
    # - min_freq       : minimum Hz to separate distinct trains (higher => stricter separation)
    # - prominence     : threshold on TTL diffs to detect rising/falling edges
    STIM_SETTINGS = {
        "stim_threshold": 0.05,
        "min_freq": 1.0,
        "prominence": 1.0,
    }
    
    # ------------------------------- RUNTIME INPUTS ------------------------------ #
    # Only three interactive inputs, as requested:
    parent_path = input("Enter parent path: ").strip(' "\'')

    if not os.path.isdir(parent_path):
        print(f"ERROR: not a directory: {parent_path}", file=sys.stderr)
        sys.exit(1)
    
    index_input = input("Enter index.csv file name [default: index.csv in parent path]: ").strip()
    if index_input:
        index_path = os.path.join(parent_path, index_input)
    else:
        index_path = os.path.join(parent_path, "index.csv")

    if not os.path.isfile(index_path):
        print(f"ERROR: index file not found: {index_path}", file=sys.stderr)
        sys.exit(1)
        
    stim_ch = input("Enter stim channel number (Default is 4): ")
    if stim_ch:
        try:
            stim_ch = int(stim_ch.strip())
        except ValueError:
            print("ERROR: stim channel must be an integer.", file=sys.stderr)
            sys.exit(1)
    else: 
        stim_ch = 4

    # ------------------------------- PIPELINE ----------------------------------- #
    # 1) Load the index describing files to process
    index = pd.read_csv(index_path, keep_default_na=False)

    # 2) Detect stim trains across files listed in the index
    detected = parse_multiple_files(
        parent_path=parent_path,
        index_df=index,
        stim_ch=stim_ch,
        stim_threshold=STIM_SETTINGS["stim_threshold"],
        min_freq=STIM_SETTINGS["min_freq"],
        prominence=STIM_SETTINGS["prominence"],
        show_progress=True
    )

    # 3) Merge detected stim info back into the original index layout
    index_with_stim = combine_with_index(index_df=index, detected_df=detected)

    # 4) Save
    out_path = os.path.join(os.path.dirname(index_path), "combined_index.csv")
    index_with_stim.to_csv(out_path, index=False)
    print(f"---> \nWrote: {out_path}")

    # For interactive runs, you might want to preview the head:
    print(index_with_stim.head().to_string(index=False))
