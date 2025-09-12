# -*- coding: utf-8 -*-

### ----------------------------- IMPORTS ----------------------------- ###
import os
import adi
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import stft
import matplotlib.pyplot as plt
import seaborn as sns
### -------------------------------------------------------------------- ###

sns.set_theme(font_scale=2)


def get_data(path, data_ch, start, stop, block):
    """
    Load signal data from a specified channel and block in a LabChart file.

    Parameters
    ----------
    path : str
        Path to the LabChart file.
    data_ch : int
        Index of the channel to extract.
    start : int
        Starting sample index.
    stop : int
        Ending sample index.
    block : int
        Block number to read from.

    Returns
    -------
    data : ndarray
        Extracted signal samples from the specified range.
    fs : float
        Sampling frequency (Hz) of the channel for the given block.
    """
    fread = adi.read_file(path)
    ch_obj = fread.channels[data_ch]
    fs = ch_obj.fs[block]
    data = ch_obj.get_data(block+1, start_sample=start, stop_sample=stop)
    return data, fs


def outlier_removal(data, percentile_threshold=[1, 95]):
    """
    Remove outliers from a signal and replace them with linear interpolation.

    Parameters
    ----------
    data : ndarray
        Input signal.
    percentile_threshold : list of float, optional
        Lower and upper percentiles for cutoff. Values outside this range
        are treated as outliers. Default is [1, 95].

    Returns
    -------
    data : ndarray
        Signal with outliers removed and replaced via linear interpolation.
    """
    nan_indices = (
        (data < np.percentile(data, percentile_threshold[0])) |
        (data > np.percentile(data, percentile_threshold[1]))
    )
    data[nan_indices] = np.nan
    data = pd.Series(data).interpolate(method='linear').values
    return data


def get_stft(data, fs, win=0.5, overlap=0.5, f_range=[2, 80]):
    """
    Compute Short-Time Fourier Transform (STFT) and return power spectrum.

    Parameters
    ----------
    data : ndarray
        Input signal.
    fs : float
        Sampling frequency in Hz.
    win : float, optional
        Window size in seconds. Default is 0.5.
    overlap : float, optional
        Fraction of overlap between windows. Default is 0.5.
    f_range : list of float, optional
        Frequency range [low, high] to extract. Default is [2, 80].

    Returns
    -------
    f : ndarray
        Frequencies (Hz) within the specified range.
    t : ndarray
        Time bins corresponding to the STFT.
    power : ndarray
        Power spectrum (magnitude squared) restricted to f_range.
    """
    f, t, zxx = stft(data, fs, nperseg=int(fs*win), noverlap=int(fs*win*overlap))
    power = np.abs(zxx)**2
    fidx = [np.argmin(np.abs(f-x)) for x in f_range]
    f = f[fidx[0]:fidx[1]]
    power = power[fidx[0]:fidx[1], :]
    return f, t, power


if __name__ == '__main__':
    # ================== SETTINGS ================== #
    main_path = r'R:\Pantelis\for analysis\_completed\LFP_awake\SST_ChR2_awake\batch2'  # Path to main directory with data and combined_index.csv
    win = 0.5         # Window size for STFT (seconds)
    overlap = 0.5     # Fraction of overlap between consecutive STFT windows (0–1)
    f_range = [10, 49]  # Frequency range [low, high] in Hz for peak frequency analysis
    outlier_threshold = [0.18, 99.9]  # Percentile thresholds for outlier removal (lower %, upper %)
    baseline_time = 2  # Length of baseline window before stim onset (seconds)
    stim_bins = [3, 7, 15, 23, 29, 33, 37, 41, 46, 51, 65, 74, 84, 94, 104]  # Bin edges for mapping raw stim frequencies
    stim_labels = [5, 10, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]  # Frequency labels (Hz) for each bin
    # ============================================== #

    # Load index file (contains metadata about recordings)
    index = pd.read_csv(os.path.join(main_path, 'combined_index.csv'), keep_default_na=False)

    # Map raw stim frequencies into discrete bins defined above
    index['stim_hz'] = index.groupby('stim_hz', group_keys=False)['stim_hz'].apply(
        pd.cut,
        bins=stim_bins,
        labels=stim_labels
    ).astype(int)

    power_ratio = []  # Store stim-to-baseline power ratios for each row
    peak_freq = []    # Store baseline peak frequency values for each row

    # ----------------- ANALYSIS LOOP ----------------- #
    for _, row in tqdm(index.iterrows(), total=len(index)):  # Iterate through all rows of index with progress bar
        row_path = os.path.join(main_path, row['folder_path'], row['file_name'])  # Path to current data file

        # --- Stim segment ---
        data, fs = get_data(row_path, row.channel_id, row['start_time'], row['stop_time'], row.block)  # Load stim segment
        data = outlier_removal(data, percentile_threshold=outlier_threshold)  # Remove outliers
        f, t, pmat = get_stft(data, fs, win=win, overlap=overlap, f_range=[row['stim_hz']-1, row['stim_hz']+1])  # STFT around stim freq
        stim_power = np.mean(pmat)  # Mean power during stimulation

        # --- Baseline segment ---
        data, fs = get_data(
            row_path, row.channel_id,
            start=row['start_time'] - int(baseline_time*fs),  # Start baseline before stim onset
            stop=row['start_time']-1,                         # End baseline at stim onset
            block=row.block
        )
        data = outlier_removal(data, percentile_threshold=outlier_threshold)  # Remove outliers
        f, t, pmat = get_stft(data, fs, win=win, overlap=overlap, f_range=[row['stim_hz']-1, row['stim_hz']+1])  # STFT around stim freq
        base_power = np.mean(pmat)  # Mean power during baseline

        # --- Peak frequency during baseline ---
        data, fs = get_data(
            row_path, row.channel_id,
            start=row['start_time'] - int(baseline_time*fs),  # Same baseline window
            stop=row['start_time']-1,
            block=row.block
        )
        data = outlier_removal(data, percentile_threshold=outlier_threshold)  # Remove outliers
        f, t, pmat = get_stft(data, fs, win=win, overlap=overlap, f_range=f_range)  # Full STFT across f_range
        psd = np.mean(pmat, axis=1)  # Average across time to get PSD
        idx = np.argmax(psd)  # Frequency index of max PSD (peak frequency)
        peak_freq.append(f[idx])  # Save peak frequency
        power_ratio.append(stim_power/base_power)  # Ratio of stim to baseline power

    # Add analysis results back into the index DataFrame
    index['power_ratio'] = power_ratio  # Add power ratio column
    index['peak_freq'] = peak_freq      # Add peak frequency column

    # ----------------- PLOTTING ----------------- #
    g = sns.relplot(
        data=index, x='stim_hz', y='power_ratio', errorbar='se',
        hue='animal_id', row='brain_region', kind='line'
    )
    for ax in g.axes.flatten():
        ax.axhline(1, linestyle='--', color='gray')  # Reference line at 1
        ax.set_title("Press 'L' in the plot window to toggle log scale")  # Add instruction to title

    g = sns.relplot(
        data=index, x='stim_hz', y='power_ratio', errorbar='se',
        row='brain_region', kind='line'
    )
    for ax in g.axes.flatten():
        ax.axhline(1, linestyle='--', color='gray')  # Reference line at 1
        ax.set_title("Press 'L' in the plot window to toggle log scale")  # Add instruction to title

    plt.show()  # Show plots
