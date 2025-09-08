# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
import adi
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from preprocess import butter_bandpass_filter, downsample
##### ------------------------------------------------------------------- #####

def get_data(path, data_ch, start, stop, block=1):
    
    fread = adi.read_file(path)
    ch_obj = fread.channels[data_ch]
    fs = ch_obj.fs[0]
    data = ch_obj.get_data(block, start_sample=start, 
                            stop_sample=stop)
    return data, fs

def outlier_removal(data, percentile_threshold=[1, 95]):
    # remove outliers
    nan_indices = (data < np.percentile(data, percentile_threshold[0])) | (data > np.percentile(data, percentile_threshold[1]))
    data[nan_indices] = np.nan

    # fill with linear interpolation
    data = pd.Series(data).interpolate(method='linear').values
    return data

def filter_data(sig, fs, frange):
    """
    Filter signal and get phase.

    Parameters
    ----------
    sig : 1d array
    fs : int
    frange : list like

    Returns
    -------
    filt_sig : 1d array, filtered signal
    phase : 1d array, instanteneous phase
    """
    # down_sig = downsample(sig, fs, new_fs)
    filt_sig = butter_bandpass_filter(sig, frange[0], frange[1], fs)
    inst_phase = np.angle(signal.hilbert(filt_sig))
    return filt_sig, inst_phase

def get_cycle_average(sig, inst_phase, fs, frange,
                      percentile=[25, 95], ncycles=2):
    
    # find cycle peaks
    extra_bins = 200
    locs,_ = signal.find_peaks(-inst_phase)
    rows = int(np.ceil(locs.shape[0]/ncycles))
    cols = int(extra_bins+fs/frange[0]*ncycles)
    
    # preallocate array
    aver_data = np.zeros((rows-1, cols))
    aver_data[:] = np.NaN
    
    # extraxt waveforms
    i=0
    for cntr in range(rows-1):
        idx = [locs[i], locs[i+ncycles]]
        aver_data[cntr,: (idx[1] - idx[0])] = sig[idx[0]:idx[1]]
        i+=ncycles
    
    # select waveforms based on amplitude and cycle duration
    counts = cols - np.sum(np.isnan(aver_data), axis=1)
    idx_cycle_dur = (counts > np.percentile(counts, percentile[0])) & (counts < np.percentile(counts, percentile[1]))
    amps = np.nanmax(aver_data, axis=1)
    idx_amp = (amps > np.percentile(amps, percentile[0])) & (amps < np.percentile(amps, percentile[1]))
    idx = idx_amp & idx_cycle_dur
    aver_wave = np.nanmean(aver_data[idx,:], axis=0)
    
    return aver_wave[:-extra_bins], amps

if __name__ == '__main__':
    
    # set path and fft settings
    main_path = r'\\rstore1\tusm_lab_maguire$\Pantelis\for analysis\LFP_awake\SST_ChR2_awake\batch2'
    percentile = [15, 99]
    baseline_time = 2
    frange = [6, 12]
    
    # get index
    index = pd.read_csv(os.path.join(main_path, 'combined_index.csv'),  keep_default_na=False)
    
    # map index frequencies to correct stim values
    index = index[(index['stim_hz']>frange[0]) & (index['stim_hz']<frange[1])]
    # add baseline
    base_index = index.copy()
    stim_index = index.copy()
    base_index['condition'] = 'baseline'
    stim_index['condition'] = 'stim' 
    base_index['start_time'] = stim_index['start_time'] - baseline_time*int(index.sampling_rate.iloc[0]) - 1
    base_index['stop_time'] = stim_index['start_time']  -1
    index = pd.concat([base_index, stim_index]).reset_index(drop=True)
    
    df_list = []
    for i,row in tqdm(index.iterrows(), total=len(index)):

        # get data, remove outliers and obtain stft
        row_path = os.path.join(main_path, row['folder_path'], row['file_name'])
    
        # get data and filter signal (add one second on either side for filtering edge effects)
        fs = int(row.sampling_rate)
        sig, _ = get_data(row_path, data_ch=row.channel_id,
                            start= int(row['start_time'] - fs), 
                            stop= int(row['stop_time'] + fs))
        
        # trim extra added and get average waveform
        frange = frange#[row['stim_hz']-1, row['stim_hz']+1] #frange #
        filt_sig, inst_phase = filter_data(sig, fs, frange=frange)
        sig = sig[fs:-fs]
        filt_sig = filt_sig[fs:-fs]
        inst_phase = inst_phase[fs:-fs]
        aver_wave, amps = get_cycle_average(filt_sig, inst_phase, fs, frange, percentile=percentile)
        
        
        # add to dataframe
        time = np.arange(len(aver_wave))/fs*1000
        df = pd.DataFrame({'animal_id':[row.animal_id]*len(aver_wave),
                           'condition':[row.condition]*len(aver_wave),
                           'stim_hz':[row.stim_hz]*len(aver_wave),
                           'time':time,
                           'aver_wave':aver_wave,
                           
                           })
        df_list.append(df)
    
    data = pd.concat(df_list).reset_index(drop=True)
    g = sns.relplot(data=data, x='time', y='aver_wave', col='animal_id', hue='condition',
                    row='stim_hz', kind='line', errorbar='se')
        
    aver_data = data.groupby(['animal_id', 'stim_hz', 'time', 'condition'], ).mean(numeric_only=True).reset_index()
    g = sns.relplot(data=aver_data, x='time', y='aver_wave', hue='condition',
                    col='stim_hz', kind='line', errorbar='se')
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    