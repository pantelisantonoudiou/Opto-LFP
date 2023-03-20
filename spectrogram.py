# -*- coding: utf-8 -*-

import os
import adi
import numpy as np
import pandas as pd
from scipy.signal import stft
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# set path
main_path = r'Z:\Pantelis\Ashley_LFP\sst_chr2'

# get index
index = pd.read_csv(os.path.join(main_path, 'combined_index.csv'))

# read data
row = index.loc[99]
fread = adi.read_file(os.path.join(main_path, row['folder_path'], row['file_name']))

# get data 
stim_ch = 0
ch_obj = fread.channels[stim_ch]
fs = ch_obj.fs[0]
time_bounds = int(1*fs)
data = ch_obj.get_data(1, start_sample=row['start_time']-time_bounds, 
                       stop_sample=row['stop_time']+time_bounds)

# remove outliers
nan_indices = (data < np.percentile(data,0.18)) | (data > np.percentile(data,99.9))
data[nan_indices] = np.nan
time = np.arange(0, data.shape[0], 1)/fs
data = pd.Series(data).interpolate(method='linear').values
data = (data - np.mean(data))/ np.std(data)


# plot data
fig, axs = plt.subplots(nrows=2)


stim_start=1
stim_end=3
rect = Rectangle((stim_start, -4), stim_end-stim_start, 8, linewidth=0,
                 facecolor='orange', alpha=0.3, zorder=10)
axs[0].add_patch(rect)
axs[0].plot(time, data, color='k')

# get power
f_bounds = [10, 80]
f, t, zxx = stft(data, fs, nperseg=int(fs/2), noverlap=int(fs/4))
power = np.abs(zxx)**2
fidx = [np.argmin(np.abs(f-x)) for x in f_bounds]
f = f[fidx[0]:fidx[1]]
power = power[fidx[0]:fidx[1],:]

# plot spectrogram
axs[1].pcolormesh(t, f, power, shading='gouraud', cmap='inferno')
axs[1].title('STFT Magnitude')
axs[1].set_ylabel('Frequency [Hz]')
axs[1].xlabel('Time [sec]')
plt.colorbar()

# plt.plot(f, np.mean(power,axis=1))
