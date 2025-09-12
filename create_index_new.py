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
    
    def __init__(self, root_dir, channel_names, drop_keywords, stim_channel_name):
        
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
        filt_index = file_index.copy()
        str_filter = "|".join(self.drop_keywords)
        mask = filt_index['adi_channel_name'].str.contains(str_filter, case=False)
        return filt_index[~mask].reset_index()
        

if __name__ == '__main__':
    
    # File contract
    # Note each file has one animal, multiple empty channels that we need to drop (number varies)
    # Each file has one stim channel and multiple blocks

    # get user input
    settings = run_index_inputs_gui()
    stim_settings = {
        "stim_threshold": 0.05,
        "min_freq": 1.0,
        "prominence": 1.0,
    }
    
    # instantiate maker class
    idx_maker = IndexMaker(
        root_dir=settings['root_dir'],
        channel_names=settings['channel_names'],
        drop_keywords=settings['drop_keywords'],
        stim_channel_name= settings['stim_channel_name'],
        )
    
    idx_maker.iterate_files(idx_maker.stim_channel_check)
    file_index = idx_maker.iterate_files(idx_maker.get_properties)
    
    # drop keywords
    filt_index = idx_maker.drop_rows(file_index)
    
    # detect stims, append to stim_df_list and create a dataframe
    # group per file and block to reduce operations
    stim_df_list = []
    group_df = filt_index.groupby(['filename', 'block'])
    for (filename, block), df in tqdm(group_df, total=len(group_df)):
        
        # parse stims and return df
        file_path = os.path.join(idx_maker.root_dir, filename + '.adicht')
        stim_id = df['stim_channel_id'].unique()[0]
        stim_df = parse_stims(
            file_path=file_path,
            stim_ch=stim_id,
            block=block,
            min_freq=stim_settings['min_freq'],
            prominence=stim_settings['prominence'],
            stim_threshold=stim_settings['stim_threshold'],
            )
        
        # add group conditions and append to list
        stim_df['filename'] = filename
        stim_df['block'] = block
        stim_df_list.append(stim_df)
    all_stim_df = pd.concat(stim_df_list, axis=0).reset_index(drop=True)
    
    # join dfs
    merged = pd.merge(filt_index, all_stim_df, on=['filename', 'block'], how='inner')
    
    



















