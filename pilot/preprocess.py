# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
import multiprocessing
from joblib import delayed, Parallel
import adi
##### ------------------------------------------------------------------- #####

# =============================================================================
#                   Basic downsampling and filtering functions
# =============================================================================

def downsample(array, fs, new_fs):
    """
    Downsamples array.

    Parameters
    ----------
    array : array, 1D signal
    fs : int, sampling frequency
    new_fs : int, new sampling frequency

    Returns
    -------
    ds_array : array, 1D downsampled signal

    """

    sample_rate = int(fs)
    ds_factor = int(sample_rate/new_fs)
    ds_array = signal.decimate(array.astype(np.float32), ds_factor, ftype='fir')
    return ds_array


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    Bandpass array using butterworth filter.

    Parameters
    ----------
    data : array, 1D signal
    lowcut : float, lower frequency cutoff
    highcut : float, upper frequency cutoff
    fs : int, sampling rate
    order : int, filter order. The default is 2

    Returns
    -------
    filt_data : array, 1D filtered signal

    """
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    filt_data = signal.filtfilt(b, a, data)
    
    return filt_data


def butter_bandpass_filter_sos(data, lowcut, highcut, fs, order=2):
    """
    Bandpass array using butterworth filter.

    Parameters
    ----------
    data : array, 1D signal
    lowcut : float, lower frequency cutoff
    highcut : float, upper frequency cutoff
    fs : int, sampling rate
    order : int, filter order. The default is 2

    Returns
    -------
    filt_data : array, 1D filtered signal

    """
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    sos  = signal.butter(order, [low, high], btype='bandpass', output='sos')
    filt_data = signal.sosfilt(sos, data)
    
    return filt_data


def cheby_bandpass(data, lowcut, highcut, fs, order=100):
    """
    cheby_bandpass(data, flow, fhigh, fs, , Rs = 100)

    Parameters
    ----------
    data :  1d ndarray, signal 
    cutoff : List [lowcut = Float, low bound frequency limit,  highcut = Float, upper bound frequency limit]
    fs : Int, sampling rate
    Rs: Int, stopband attenuation in Db, Optional, Default value = 100.

    Returns
    -------
    y : 1d ndarray, filtered signal 

    """
    Fn = fs/2                                          # Nyquist Frequency (Hz)
    Wp = [lowcut/Fn,   highcut/Fn]                         # Passband Frequency (Normalised)
    Ws = [(lowcut-1)/Fn,   (highcut+1)/Fn]                 # Stopband Frequency (Normalised)
    Rp = 1                                             # Passband Ripple (dB)   
    Rs = order
    
    # Get Filter Order
    n, Ws = signal.cheb2ord(Wp,Ws,Rp,Rs);
    
    # Design Filter
    sos = signal.cheby2(n,Rs,Ws, btype='bandpass', output='sos')

    # filter data
    y = signal.sosfiltfilt(sos,data)
    return y  