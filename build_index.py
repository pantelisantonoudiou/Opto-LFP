# -*- coding: utf-8 -*-
# =============================================================================
#                                 Imports
# =============================================================================
import os
import numpy as np
import pandas as pd
import adi
from index_inputs_gui import run_index_inputs_gui
from tqdm import tqdm
# =============================================================================
# =============================================================================

def parse_stims(file_path, stim_ch, block, min_freq=1, prominence=1, stim_threshold=2, ttl_height=5):
    """
    Parse stimulation trains from a LabChart file.

    Detects stimulation trains in a specified channel, reconstructs a TTL-like
    signal, extracts train start/stop sample indices, estimates the train
    frequency, and returns a summary DataFrame.

    Parameters
    ----------
    file_path : str, Path to the LabChart file to be parsed.
    stim_ch : int, Index of the stimulation channel (1-based in this implementation).
    block : int, Block number to read from the LabChart file.
    min_freq : float, optional (default=1),  Minimum frequency (Hz) used to separate distinct trains.
        Internally converted to a minimum gap (samples) between trains.
    prominence : float, optional (default=1), Threshold applied to the TTL diff to detect rising/falling edges.
    stim_threshold : float, optional (default=2), Amplitude threshold applied to the stim channel to create a boolean TTL-like signal.
    ttl_height : float, optional (default=5), Height assigned to the TTL signal when above threshold.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns:
        - stim_hz   : int
        - start_time: int (sample index)
        - stop_time : int (sample index)
        - block     : int
    """
    
    # Read LabChart file
    fread = adi.read_file(file_path)

    # Select stimulation channel
    ch_obj = fread.channels[stim_ch]
    stim = ch_obj.get_data(block + 1)  # ADI blocks are 1-based
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

    # Build output dataframe
    df = pd.DataFrame({
        'stim_hz': freqs,
        'start_time': start - time_from_first_peak,
        'stop_time': stop + time_from_first_peak,
        'block': block
    })
    return df


class IndexMaker():
    """
    Build and manipulate an index of channels and stimulation metadata for LabChart ADI files.

    This helper class:
      1) Scans a root directory for `.adicht` files.
      2) Validates that each file contains exactly one stimulation channel matching
         a user-provided name (substring match, case-insensitive).
      3) Extracts per-file / per-channel / per-block properties into a tidy DataFrame.
      4) Filters out unwanted channels using user-specified keywords.
      5) Provides an iterator utility to apply a single function across all files.

    Parameters
    ----------
    root_dir : str
        Directory containing LabChart `.adicht` files.
    channel_names : list[str]
        List of channel search terms (e.g., brain region names) to include. Matches are
        done via substring search (case-insensitive).
    drop_keywords : list[str]
        List of substrings used to drop channels later (e.g., ["empty", "unused"]).
    stim_channel_name : str
        Substring that identifies the stimulation channel (case-insensitive).

    Attributes
    ----------
    root_dir : str
        Root directory scanned for `.adicht` files.
    selected_channels : list[str]
        Search terms for channels/brain regions to include.
    drop_keywords : list[str]
        Substrings used to filter out channels in `drop_rows`.
    stim_name : str
        Substring used to locate the stimulation channel.
    adi_files : list[str]
        Filenames (not full paths) of detected `.adicht` files.

    Notes
    -----
    - No logic within methods is altered; comments and docstrings only.
    - Channel matching uses Pandas `str.contains` with case-insensitive search.
    """

    def __init__(self, root_dir, channel_names, drop_keywords, stim_channel_name):
        """Initialize configuration and discover `.adicht` files in the root directory."""
        # get settings
        self.root_dir = root_dir
        self.selected_channels = channel_names
        self.drop_keywords = drop_keywords
        self.stim_name = stim_channel_name
        
        # get adi files 
        files = os.listdir(self.root_dir)
        self.adi_files = [file for file in files if file[-6:] == 'adicht']
        
        if len(self.adi_files) == 0:
            raise Exception('No labchart files detected. Aborting operation')
    
    def stim_channel_check(self, adi_path):
        """
        Verify that exactly one stimulation channel exists in a given ADI file.

        Parameters
        ----------
        adi_path : str
            Full path to a `.adicht` file.

        Raises
        ------
        Exception
            If zero or more than one stimulation channels are found.

        Returns
        -------
        None
            Used for side-effect validation only.
        """
        # read channel and get stim containing channels
        adi_obj = adi.read_file(adi_path)
        channels = pd.Series([ch.name for ch in adi_obj.channels]).str.lower()
        n_stim_channels = sum(channels.str.contains(self.stim_name))
        if n_stim_channels == 0:
            raise Exception(f'Channel name was not found in {adi_path}, instead got: {channels}')
        elif n_stim_channels > 1:
            raise Exception(f'Found {n_stim_channels} stim channels, in file {adi_path}. Only one is allowed.')
        else:
            del adi_obj
        return None
            
    def get_properties(self, adi_path):
        """
        Extract per-channel and per-block properties from a single ADI file.

        For each user-requested channel term (in `self.selected_channels`), this
        scans the file's channels for substring matches, then emits one row per
        (matched channel, block) with metadata required to later parse stims and
        join results.

        Parameters
        ----------
        adi_path : str
            Full path to a `.adicht` file.

        Returns
        -------
        pandas.DataFrame or None
            DataFrame with columns:
              - filename : str (basename without extension)
              - sampling_rate : int (Hz)
              - stim_channel_id : int (index of stim channel in the ADI file)
              - adi_channel_name : str (matched channel name, lowercase)
              - brain_region : str (the search term that matched)
              - channel_id : int (index of the matched channel)
              - block : int (block number)
            Returns None if no properties are detected.
        """
        # read file and get properties for each block
        # only include channels found
        adi_obj = adi.read_file(adi_path)
        channels = pd.Series([ch.name for ch in adi_obj.channels]).str.lower()
        blocks = adi_obj.n_records
        sampling_rate = int(1/adi_obj.records[0].tick_dt)
        filename = os.path.basename(adi_path).replace('.adicht', '')
        stim_ch_id = np.where(channels.str.contains(self.stim_name))[0][0]

        # iterate over user search channels (or brain regions)
        properties_list = []
        for sel_channel in self.selected_channels:
            
            # iterate over found channels and append properties to list
            ch_idx = np.where(channels.str.contains(sel_channel))[0]
            for idx in ch_idx:
                properties = {'filename': filename,
                              'sampling_rate':sampling_rate,
                              'stim_channel_id':stim_ch_id,
                              'adi_channel_name': channels.iloc[idx],
                              'brain_region': sel_channel,
                              'channel_id': idx}
                
                # iterate over blocks
                for block in range(blocks):
                    block_properties = properties.copy()
                    block_properties['block'] = block
                    properties_list.append(block_properties)
        
        # if properties detected return dataframe
        if len(properties_list) > 0:
            return pd.DataFrame(properties_list)
        
        del adi_obj
        return None
            
    def iterate_files(self, func):
        """
        Apply a single callable to each ADI file and collect non-None outputs.

        The callable should accept a single argument: the full path to the file.
        If the callable returns a DataFrame for some files, those frames are concatenated
        and returned; otherwise returns None.

        Parameters
        ----------
        func : Callable[[str], Any]
            Function applied to each file path. Often one of:
              - `self.stim_channel_check`  (validation, returns None)
              - `self.get_properties`      (returns a DataFrame per file)

        Returns
        -------
        pandas.DataFrame or None
            Concatenated output of all non-None returns from `func`.
            Returns None if nothing is collected.
        """
        # iterate over files and apply function
        output_list = []
        for adi_file in tqdm(self.adi_files):
            output = func(os.path.join(self.root_dir, adi_file))
            if output is not None:
                output_list.append(output)
                
        if len(output_list) > 0:
            return pd.concat(output_list).reset_index(drop=True)
        else:
            return None
        
    def drop_rows(self, file_index):
        """
        Drop rows whose `adi_channel_name` contains any of `self.drop_keywords`.

        Parameters
        ----------
        file_index : pandas.DataFrame
            DataFrame produced by `get_properties` (or a concatenation thereof).

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame with rows containing drop keywords removed.
            Index is reset after filtering.
        """
        filt_index = file_index.copy()
        str_filter = "|".join(self.drop_keywords)
        mask = filt_index['adi_channel_name'].str.contains(str_filter, case=False)
        return filt_index[~mask].reset_index()

def build_stim_index(filt_index, idx_maker, stim_settings):
    """
    Detect stimulation trains for each (filename, block) group and merge results
    with the provided filtered index.

    Parameters
    ----------
    filt_index : pandas.DataFrame
        Filtered index of channels from IndexMaker.get_properties + drop_rows.
        Must include at least columns: ['filename', 'block', 'stim_channel_id'].
    idx_maker : IndexMaker
        Instance of IndexMaker with valid root_dir.
    stim_settings : dict
        Dictionary with stim parsing parameters, e.g.:
        {
            "stim_threshold": 0.05,
            "min_freq": 1.0,
            "prominence": 1.0,
        }

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame combining channel properties and stimulation metadata.
    """
    stim_df_list = []

    # group per file and block to reduce redundant stim parsing
    group_df = filt_index.groupby(['filename', 'block'])
    for (filename, block), df in tqdm(group_df, total=len(group_df)):
        # full path to .adicht file
        file_path = os.path.join(idx_maker.root_dir, filename + '.adicht')

        # each group should have a single stim_channel_id
        stim_id = df['stim_channel_id'].unique()[0]

        # parse stim trains
        stim_df = parse_stims(
            file_path=file_path,
            stim_ch=stim_id,
            block=block,
            min_freq=stim_settings['min_freq'],
            prominence=stim_settings['prominence'],
            stim_threshold=stim_settings['stim_threshold'],
        )

        # annotate with grouping metadata
        stim_df['filename'] = filename
        stim_df['block'] = block
        stim_df_list.append(stim_df)

    # combine stim data across all groups
    all_stim_df = pd.concat(stim_df_list, axis=0).reset_index(drop=True)

    # merge with channel index
    merged = pd.merge(filt_index, all_stim_df, on=['filename', 'block'], how='inner')
    return merged

if __name__ == '__main__':
    
    """
    Main execution pipeline for building a stimulation index from LabChart ADI files.

    File/Data Contract
    ------------------
    - Each `.adicht` file corresponds to one animal.
    - Each file may contain multiple blocks (recordings).
    - Each file must contain exactly one stimulation channel (e.g. opto TTL).
    - Files may contain multiple empty/unwanted channels, removed via `drop_keywords`.

    Workflow
    --------
    - Collect user-defined settings via GUI.
    - Initialize IndexMaker and discover `.adicht` files.
    - Validate that each file has exactly one stim channel.
    - Extract per-file, per-block channel properties.
    - Drop channels matching user-defined keywords.
    - Parse stimulation trains for each file/block.
    - Merge channel properties with stim metadata into final index.
    """

    # get user input
    settings = run_index_inputs_gui()
    if settings is None:
        raise Exception('No User Inputs. Aborting operation.')
    
    stim_settings = {
        "stim_threshold": 0.05,
        "min_freq": 1.0,
        "prominence": 1.0,
    }
    
    # instantiate index maker class
    idx_maker = IndexMaker(
        root_dir=settings['root_dir'],
        channel_names=settings['channel_names'],
        drop_keywords=settings['drop_keywords'],
        stim_channel_name= settings['stim_channel_name'],
        )
    
    # check for detection of one stim channel per file
    print('Stim Check...')
    idx_maker.iterate_files(idx_maker.stim_channel_check)
    
    # create an index file for each block/recording
    print('...')
    file_index = idx_maker.iterate_files(idx_maker.get_properties)
    filt_index = idx_maker.drop_rows(file_index)
    
    # build final stim index
    merged = build_stim_index(filt_index, idx_maker, stim_settings)
    
