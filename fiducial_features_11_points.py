import neurokit2 as nk
import numpy as np
from sklearn import preprocessing

def get_fiducial_features(ecg_signal):
    
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=1000)
    signal_cwt, waves_cwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=1000,method="cwt",show=True,show_type='all')
    
    l=len(waves_cwt['ECG_Q_Peaks'])
    
    feature_array=[]
    for i in range(l-1):

        vector=[]
        vector.append(waves_cwt['ECG_P_Onsets'][i])
        vector.append(waves_cwt['ECG_P_Peaks'][i])
        vector.append(waves_cwt['ECG_P_Offsets'][i])


        vector.append(waves_cwt['ECG_R_Onsets'][i])
        vector.append(waves_cwt['ECG_Q_Peaks'][i])

        vector.append(rpeaks['ECG_R_Peaks'][i])

        vector.append(waves_cwt['ECG_S_Peaks'][i])
        vector.append(waves_cwt['ECG_R_Offsets'][i])


        vector.append(waves_cwt['ECG_T_Onsets'][i])
        vector.append(waves_cwt['ECG_T_Peaks'][i])
        vector.append(waves_cwt['ECG_T_Offsets'][i])

        feature_array.append(vector)
        
    feature_array=np.vstack(feature_array)
    
    feature_array[feature_array==np.nan]
    s=np.isnan(feature_array)
    feature_array[s]=0 
    
    return feature_array
   

    

