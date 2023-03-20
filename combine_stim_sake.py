# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import adi
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
##### ------------------------------------------------------------------- #####


def parse_stims(file_path, lfp_ch, stim_ch, 
                 prominence=1):
    
    # read labchart file
    fread = adi.read_file(file_path)
    
    # get stim 
    ch_obj = fread.channels[stim_ch]
    stim = ch_obj.get_data(1)
    fs = ch_obj.fs[0]

    # detect trains
    locs,_ = signal.find_peaks(np.gradient(stim), prominence=prominence)

    # recreate ttl
    # ttl = fread.channels[ttl_ch].get_data(1)
    min_freq = int(fs/3)
    start = locs[np.append(True, np.diff(locs) > min_freq)]
    stop = locs[np.append(np.diff(locs) > min_freq, True)]
    ttl = np.zeros(stim.shape[0])
    for x,y in zip(start,stop):
        ttl[x:y] = 5

    # find pulse start and stop
    start,_ = signal.find_peaks(np.gradient(ttl), prominence=prominence)
    stop,_ = signal.find_peaks(np.gradient(-ttl), prominence=prominence)
    
    # find train frequency
    freqs = []
    for x1, x2 in zip(start, stop):
        train = locs[(locs>=x1) & (locs<x2)]
        iei = np.median(np.diff(train))/fs
        freq = int(np.ceil(1/iei))
        freqs.append(freq)

    time_from_first_peak = (fs/np.array(freqs)).astype(int)

    # get animal id
    animal_id = fread.channels[lfp_ch].name.split('-')
    animal_id = '-' + animal_id[1] + '-'

    # create output dataframe
    df = pd.DataFrame({'animal_id': np.repeat([animal_id], len(freqs)),
                       'stim_hz':freqs,
                       'start_time':start - time_from_first_peak,
                       'stop_time':stop + time_from_first_peak,
                       })
    
    return df


def parse_multiple_files(main_path, index):
    """
    Detect stims from all files in index.

    Parameters
    ----------
    main_path : str
    index : pandas df

    Returns
    -------
    df : pandas df

    """

    df_list = []
    for i, row in tqdm(index.iterrows(), total=len(index)):
        file_path = os.path.join(main_path, row.folder_path, row.file_name)
        df = parse_stims(file_path, lfp_ch=0, stim_ch=1)
        df_list.append(df)
    df = pd.concat(df_list, axis=0)
    return df


if __name__ == '__main__':
    
    # set path
    main_path = r'Z:\Pantelis\Ashley_LFP\sst_chr2'
    
    # get index
    index = pd.read_csv(os.path.join(main_path, 'index.csv'))
    
    # parse all animals
    df = parse_multiple_files(main_path, index)
    
    # combine
    df_list = []
    for i,row in index.iterrows():
        # find matching rows
        idx = (df['animal_id'] == row.animal_id) & (df['start_time']>= row.start_time) \
            & (df['stop_time']<= row.stop_time)    
        match = df[idx]
        row = pd.DataFrame(dict(zip(row.index, row.values)), index=[i])
        row = match.set_index('animal_id').combine_first(row.set_index('animal_id'))
        df_list.append(row.reset_index())
        
    index_df = pd.concat(df_list, axis=0)
    col_order = index.columns.tolist() + ['stim_hz']
    index_df = index_df[col_order]
    index_df.to_csv(os.path.join(main_path,'combined_index.csv'), index=False)














