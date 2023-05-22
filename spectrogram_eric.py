# -*- coding: utf-8 -*-

import os
import adi
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import stft
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


def get_data(path, data_ch, start, stop, block):
    
    fread = adi.read_file(path)
    ch_obj = fread.channels[data_ch]
    fs = ch_obj.fs[block]
    data = ch_obj.get_data(block+1, start_sample=start, 
                            stop_sample=stop)
    return data, fs
    

def outlier_removal(data, percentile_threshold=[1, 95]):
    # remove outliers
    nan_indices = (data < np.percentile(data, percentile_threshold[0])) | (data > np.percentile(data, percentile_threshold[1]))
    data[nan_indices] = np.nan

    # fill with linear interpolation
    data = pd.Series(data).interpolate(method='linear').values
    return data


def get_stft(data, fs, win=0.5, overlap=0.5, f_range=[2, 80]):

    f, t, zxx = stft(data, fs, nperseg=int(fs*win), noverlap=int(fs*win*overlap))
    power = np.abs(zxx)**2
    fidx = [np.argmin(np.abs(f-x)) for x in f_range]
    f = f[fidx[0]:fidx[1]]
    power = power[fidx[0]:fidx[1],:]
    return f, t, power

if __name__ == '__main__':
    
    # set path and fft settings
    main_path = r'C:\Users\pante\Desktop\eric_opto'
    win = 0.5
    overlap = 0.5
    f_range = [10, 49]
    outlier_threshold = [0.18, 99.9]
    baseline_time = 30
    
    # get index
    index = pd.read_csv(os.path.join(main_path, 'combined_index.csv'), keep_default_na=False)
    index = index.fillna('')
                         
    # map index frequencies to correct stim values
    index['stim_hz'] = index.groupby('stim_hz', group_keys=False)['stim_hz'].apply(pd.cut, bins=[3, 7, 15, 23, 29, 33, 37, 41, 46, 51, 65, 74, 84, 94, 104], 
                                                                                 labels=[5, 10, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]).astype(int)
    # power_ratio = []
    # for i,row in tqdm(index.iterrows(), total=len(index)):

    #     # get data, remove outliers and obtain stft
    #     row_path = os.path.join(main_path, row['folder_path'], row['file_name'])

    #     # get stim
    #     data, fs = get_data(row_path, 
    #                         data_ch=row.channel_id,
    #                         start=row['start_time'], 
    #                         stop=row['stop_time'],
    #                         block=row.block)
    #     data = outlier_removal(data, percentile_threshold=outlier_threshold)
    #     f, t, pmat = get_stft(data, fs, win=win, overlap=overlap, 
    #                           f_range=[row['stim_hz']-1, row['stim_hz']+1])
    #     stim_power = np.mean(pmat)
        
    #     # get baseline
    #     data, fs = get_data(row_path, 
    #                         data_ch=row.channel_id,
    #                         start=row['start_time'] - int(baseline_time*fs), 
    #                         stop=row['start_time']-1,
    #                         block=row.block)
    #     data = outlier_removal(data, percentile_threshold=outlier_threshold)
    #     f, t, pmat = get_stft(data, fs, win=win, overlap=overlap, 
    #                           f_range=[row['stim_hz']-1, row['stim_hz']+1])
    #     base_power = np.mean(pmat)
    #     power_ratio.append(stim_power/base_power)  
    # index['power_ratio'] = power_ratio


# sns.catplot(data=index, x='treatment', y='power_ratio', kind='bar', errorbar='se') #


fs = int(index.sampling_rate.iloc[0])
# add baseline
base_index = index.copy()
stim_index = index.copy()
base_index['condition'] = 'pre'
stim_index['condition'] = 'stim' 
base_index['start_time'] = stim_index['start_time'] - baseline_time*fs - 1
base_index['stop_time'] = stim_index['start_time'] - fs - 1
# stim_index = stim_index[stim_index['start_time'] > 5*60*fs]
index = pd.concat([base_index, stim_index]).reset_index(drop=True)

power_list = []
for i,row in tqdm(index.iterrows(), total=len(index)):

    # get data, remove outliers and obtain stft
    row_path = os.path.join(main_path, row['folder_path'], row['file_name'])


    # get stim
    data, fs = get_data(row_path, 
                        data_ch=row.channel_id,
                        start=row['start_time'], 
                        stop=row['stop_time'],
                        block=row.block)
    data = outlier_removal(data, percentile_threshold=outlier_threshold)
    f, t, pmat = get_stft(data, fs, win=win, overlap=overlap, 
                          f_range=[row['stim_hz']-2, row['stim_hz']+2])
    power = np.mean(pmat)
    power_list.append(power)
index['power'] = power_list    
    

sns.catplot(data=index, x='stop_time', y='power', hue='condition', kind='bar', errorbar='se') #

        
