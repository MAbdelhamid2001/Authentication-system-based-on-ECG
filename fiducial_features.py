import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def pan_tompkins(ecg_signal, fs=1000):
    # Bandpass filter the ECG signal
    ecg_signal_filtered = butter_bandpass_filter(ecg_signal, 2, 40, fs)

    # Differentiate the signal
    diff_signal = np.diff(ecg_signal_filtered)

    # Square the signal
    squared_signal = diff_signal ** 2

    # Integrate the signal
    integrated_signal = np.convolve(squared_signal, np.ones(fs // 8))[:len(ecg_signal)]
# ######
    # Find the QRS complex peaks using a threshold-based approach
    qrs_peaks, _ = find_peaks(integrated_signal, height=0.35*np.max(integrated_signal), distance=0.2*fs)

    # Calculate the fiducial features
    qrs_onsets = []
    qrs_offsets = []
    p_peaks = []
    t_peaks = []
    rr_interval_low = int(fs * 0.2)
    rr_interval_high = int(fs * 1)
    
    for qrs_peak in qrs_peaks:
        # Find the QRS onset
        # Find the QRS onset and offset
        qrs_onset = qrs_peak - int(rr_interval_low * 0.5)
        qrs_offset = qrs_peak + int(rr_interval_high * 0.5)
        qrs_onsets.append(qrs_onset)
        qrs_offsets.append(qrs_offset)

#         # Find the QRS onset
#         for i in range(qrs_peak, 0, -1):
#             if integrated_signal[i] < integrated_signal[qrs_peak] * 0.15:
#                 qrs_onsets.append(i)
#                 break
#         # Find the QRS offset
#         for i in range(qrs_peak, len(ecg_signal)):
#             if integrated_signal[i] < integrated_signal[qrs_peak] * 0.2:
#                 qrs_offsets.append(i)
#                 break
                
                
        # Find the P wave peak
        p_peak, _ = find_peaks(ecg_signal_filtered[qrs_onsets[-1]:qrs_peak], height=0.1*np.max(ecg_signal_filtered), distance=0.1*fs)
        if len(p_peak) > 0:
            p_peaks.append(p_peak[-1] + qrs_onsets[-1])
        else:
            p_peaks.append(None)
        # Find the T wave peak
        t_peak, _ = find_peaks(-ecg_signal_filtered[qrs_peak:qrs_offsets[-1]], height=-0.1*np.max(ecg_signal_filtered), distance=0.2*fs)
        if len(t_peak) > 0:
            t_peaks.append(t_peak[0] + qrs_peak)
        else:
            t_peaks.append(None)

            
    feature_array = []
    ecg_signal_filtered=ecg_signal_filtered.copy()
    for i in range(len(qrs_peaks)):
        feature_vector = []
        # Add QRS complex features
        feature_vector.append(ecg_signal_filtered[qrs_peaks[i]])
        feature_vector.append(integrated_signal[qrs_peaks[i]])
        feature_vector.append(qrs_offsets[i] - qrs_onsets[i])
        feature_vector.append(qrs_onsets[i] - qrs_onsets[0])

        if p_peaks[i] is not None:
            feature_vector.append(ecg_signal_filtered[p_peaks[i]])
            feature_vector.append(qrs_peaks[i] - p_peaks[i])
        else:
            feature_vector += [None, None]
        # Add T wave features
        if t_peaks[i] is not None:
            feature_vector.append(ecg_signal_filtered[t_peaks[i]])
            feature_vector.append(t_peaks[i] - qrs_peaks[i])
        else:
            feature_vector += [None, None]
        feature_array.append(feature_vector)            
            
            
    return feature_array