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


def parse_stims(file_path, lfp_ch, stim_ch, block, min_freq=2,
                 prominence=1, stim_threshold=2):
    
    ttl_height = 5
    # read labchart file
    fread = adi.read_file(file_path)
    
    # get stim 
    ch_obj = fread.channels[stim_ch]
    stim = ch_obj.get_data(block+1)
    fs = ch_obj.fs[block]

    # detect trains (make stim boolean)
    stim_bool = (stim > stim_threshold)*ttl_height
    locs = np.where(np.diff(stim_bool) > prominence)[0] + 1
    
    # recreate ttl
    min_freq = int(fs/min_freq)
    start = locs[np.append(True, np.diff(locs) > min_freq)]
    stop = locs[np.append(np.diff(locs) > min_freq, True)]
    ttl = np.zeros(stim.shape[0])
    for x,y in zip(start,stop):
        ttl[x:y] = ttl_height
        
    # find pulse start and stop
    ttl_diff = ttl[1:]-ttl[:-1]
    start = np.where(ttl_diff > prominence)[0] + 1
    stop = np.where(-ttl_diff > prominence)[0] + 1
    
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
                       'block':block
                       })

    return df


def parse_multiple_files(main_path, index, stim_ch=12):
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
        df = parse_stims(file_path, lfp_ch=row.channel_id, stim_ch=stim_ch, 
                         block=row.block, stim_threshold=2)
        df['file_id'] = row.file_id
        df_list.append(df)
    df = pd.concat(df_list, axis=0)
    return df


if __name__ == '__main__':
    
    # set path
    main_path = r'C:\Users\pante\Desktop\eric_opto'
    stim_ch = 1
    
    # get index
    index = pd.read_csv(os.path.join(main_path, 'index.csv'), keep_default_na=False)
    
    # select only one channel per file
    # parse all animals
    df = parse_multiple_files(main_path, index, stim_ch=stim_ch)
    
    # combine
    df_list = []
    for i,row in index.iterrows():
        # find matching rows 
        idx = (df['animal_id'] == row.animal_id) & (df['block']== row.block) & (df['file_id']== row.file_id) \
            & (df['start_time']>= row.start_time) \
            & (df['stop_time']<= row.stop_time)    
        match = df[idx]
        row = pd.DataFrame(dict(zip(row.index, row.values)), index=[i])
        row = match.set_index('animal_id').combine_first(row.set_index('animal_id'))
        df_list.append(row.reset_index())
        
    index_df = pd.concat(df_list, axis=0)
    col_order = index.columns.tolist() + ['stim_hz']
    index_df = index_df[col_order]
    index_df.to_csv(os.path.join(main_path,'combined_index.csv'), index=False)














