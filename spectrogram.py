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


def get_data(path, data_ch, start, stop, block=1):
    
    fread = adi.read_file(path)
    ch_obj = fread.channels[data_ch]
    fs = ch_obj.fs[0]
    data = ch_obj.get_data(block, start_sample=start, 
                            stop_sample=stop)
    return data, fs
    

def outlier_removal(data,  percentile_threshold=[1, 95]):
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
    main_path = r'Z:\Pantelis\Ashley_LFP\sst_chr2'
    win = 0.5
    overlap = 0.5
    f_range = [10, 49]
    outlier_threshold = [0.18, 99.9]
    baseline_time = 1
    
    # get index
    index = pd.read_csv(os.path.join(main_path, 'combined_index.csv'))
    index = index[~index['animal_id'].isin(['-mouse1,s5pos1-','-mouse2,s2pos1-', '-mouse2,s4pos2-']) ]
    
    # map index frequencies to correct stim values
    index = index[index['stim_hz']>4]
    idx = (index['stim_hz']>=43) & (index['stim_hz']<=47)
    index.loc[idx,'stim_hz'] = 45
    idx = (index['stim_hz']>=58) & (index['stim_hz']<=61)
    index.loc[idx,'stim_hz'] = 60
    print(np.sort(index['stim_hz'].unique()))
    
    # select slice for analysis
    plot_index = index#[index['animal_id'].isin([#'-mouse1,s5pos1-', '-mouse1,s5pos2-',
                                                 # '-mouse2,s2pos1-', '-mouse2,s2pos2-',
                                                # '-mouse2,s4pos1-', '-mouse2,s4pos2-'
                                                # ]) ]
    
    power_ratio = []
    peak_freq = []
    for i,row in tqdm(plot_index.iterrows(), total=len(plot_index)):
        
        # get data, remove outliers and obtain stft
        row_path = os.path.join(main_path, row['folder_path'], row['file_name'])

        
        # get stim
        data, fs = get_data(row_path, 
                            data_ch=row.channel_id,
                            start=row['start_time'], 
                            stop=row['stop_time'])
        data = outlier_removal(data, percentile_threshold=outlier_threshold)
        f, t, pmat = get_stft(data, fs, win=win, overlap=overlap, 
                              f_range=[row['stim_hz']-1, row['stim_hz']+1])
        stim_power = np.mean(pmat)
        
        # get baseline
        data, fs = get_data(row_path, 
                            data_ch=row.channel_id,
                            start=row['start_time'] - int(baseline_time*fs), 
                            stop=row['start_time']-1)
        data = outlier_removal(data, percentile_threshold=outlier_threshold)
        f, t, pmat = get_stft(data, fs, win=win, overlap=overlap, 
                              f_range=[row['stim_hz']-1, row['stim_hz']+1])
        base_power = np.mean(pmat)
        
        # get peak freq during baseline
        data, fs = get_data(row_path, 
                            data_ch=row.channel_id,
                            start=row['start_time'] - int(baseline_time*fs), 
                            stop=row['start_time']-1)
        data = outlier_removal(data, percentile_threshold=outlier_threshold)
        f, t, pmat = get_stft(data, fs, win=win, overlap=overlap, 
                              f_range=f_range)
       
        psd = np.mean(pmat, axis=1)
        plt.plot(f,psd)
        idx = np.argmax(psd)
        peak_freq.append(f[idx])
        power_ratio.append(stim_power/base_power)
        
    plot_index['power_ratio'] = power_ratio
    plot_index['peak_freq'] = peak_freq

plot_data = plot_index.groupby(['animal_id', 'stim_hz'], ).mean(numeric_only=True).reset_index()
sns.set(font_scale=2)
plt.figure()
sns.lineplot(data=plot_data, x='stim_hz', y='power_ratio', errorbar=None, hue='animal_id',  alpha=.5) #color='grey',
sns.lineplot(data=plot_data, x='stim_hz', y='power_ratio', errorbar='se',color ='black')

sns.relplot(data=plot_data, x='stim_hz', y='power_ratio', errorbar='se', hue='animal_id', col='animal_id', kind='line')

sns.catplot(data=plot_data, x='animal_id', y='peak_freq', errorbar='se', kind='box')
        
        

    
    
    
        # data = (data - np.mean(data))/ np.std(data)
    

    
    
#     # plot data
#     fig, axs = plt.subplots(nrows=2)
#     time = np.arange(0, data.shape[0], 1)/fs
#     stim_start=1
#     stim_end=3
#     rect = Rectangle((stim_start, -4), stim_end-stim_start, 8, linewidth=0,
#                      facecolor='orange', alpha=0.3, zorder=10)
#     axs[0].add_patch(rect)
#     axs[0].plot(time, data, color='k')
    
    
#     # plot spectrogram
#     axs[1].pcolormesh(t, f, power, shading='gouraud', cmap='inferno')
#     axs[1].title('STFT Magnitude')
#     axs[1].set_ylabel('Frequency [Hz]')
#     axs[1].xlabel('Time [sec]')
#     plt.colorbar()

# # plt.plot(f, np.mean(power,axis=1))
