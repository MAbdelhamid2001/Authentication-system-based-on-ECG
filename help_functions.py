import numpy as np
from scipy.signal import butter,filtfilt
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
import scipy.io
import scipy
import matplotlib.pyplot as plt


def butter_bandbass_filter(Input_signal,low_cutoff,high_cutoff,sampling_rate,order=4):
    nyq=0.5*sampling_rate #nyquist sampling
    low=low_cutoff/nyq
    high=high_cutoff/nyq
    
    numerator,denominator=butter(order,[low,high],btype='band',output='ba',analog=False,fs=None)
    filtered=filtfilt(numerator,denominator,Input_signal)
    
    return filtered


from scipy.signal import find_peaks
def ecg_segmentation(ecg_signal, fs=1000, threshold=0.5):
    # Find R-peaks using a threshold-based approach
    peaks, _ = find_peaks(ecg_signal, height=threshold)

    
    # Calculate the RR intervals
    rr_intervals = np.diff(peaks) / fs

#     # Plot the ECG signal and detected R-peaks
#     time = np.arange(len(ecg_signal)) / fs
#     plt.figure(figsize=(12, 6))
#     plt.plot(time, ecg_signal, 'b', label='ECG Signal')
#     plt.plot(time[peaks], ecg_signal[peaks], 'ro', label='R-Peaks')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.title('ECG Signal Segmentation')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

    return peaks, rr_intervals

def extract_ecg_segments(ecg_signal, r_peaks, fs=1000, window_size=0.2):
    # Calculate the window size in samples
    window_size_samples = int(window_size * fs)

    # Initialize an empty array to store the segments
    segments = []

    # Loop over the R-peaks and extract the corresponding segments
    for r_peak in r_peaks:
        start = r_peak - window_size_samples // 2
        end = r_peak + window_size_samples // 2
        segment = ecg_signal[start:end]
        segments.append(segment)
#     return np.array(segments)       
#######################
    lens=[len(s) for s in segments]
    max_len=max(lens)

    new_segements=[]
    for s in segments:
        if len(s)==max_len:
            new_segements.append(s)
        else:
            zeros_size=max_len-len(s)
            s=np.append(s,np.zeros(zeros_size))
            new_segements.append(s)
############################    
    
    
    return np.array(new_segements)

def preprocess_using_ACDCT(filtered_signal):
    sig=np.array(filtered_signal)
    AC=sm.tsa.acf(sig,nlags=1000)
    s=AC[0:100]
    DCT=scipy.fftpack.dct(s,type=2)
    
    return DCT


from fiducial_features import pan_tompkins
def preprocess_using_fiducial(filtered_signal):
    
    features=pan_tompkins(filtered_signal)
    return features

def preprocess_using_wavelet(filtered_signal):  
    import pywt
    from pywt import wavedec ,waverec

    wavelet = 'db4'
    level = 4
    coeffs = pywt.wavedec(filtered_signal, wavelet, level=level)
    
    for i in range(1, level):
        coeffs[i] = np.zeros_like(coeffs[i])

    res = pywt.waverec(coeffs, 'db4')
    return res

def preprocessing(signal,type_):
    
    """
    # type_=1:use wavelet
    # type_=2:use fiducial_features
    # type_=3:use AC/DCT
    
    """
#     butterworth filter 2-40  or 2 -50 
#     sr=1000
    filtered_signal=butter_bandbass_filter(signal,low_cutoff=2,high_cutoff=40,sampling_rate=1000,order=4)
    
#     Features extraction using AC/DCT 
    
    if type_==1:
        features=preprocess_using_wavelet(filtered_signal)
    elif type_==2:
        features=preprocess_using_fiducial(filtered_signal)
    elif type_==3:
        features=preprocess_using_ACDCT(filtered_signal)
    
    return features

