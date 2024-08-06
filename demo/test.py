# Author: Thel'Vadamee
# Date: 2024/7/29 10:25

import numpy as np
import os
import scipy.io as scio
import neurokit2 as nk

# data_root = r'../DE_Whole'
#
# filelist = os.listdir(data_root)
# print(filelist)
#
# demo_pth_ECG = r'../DE_Whole/1-DE_Whole-ECG.npy'
# demo_pth_EEG = r'../DE_Whole/1-DE_Whole-EEG.npy'
# data_ECG = np.load(demo_pth_ECG)
# data_EEG = np.load(demo_pth_EEG)
# print('=============ECG=============')
# print(type(data_ECG))
# print(data_ECG.shape)
# # print(data_ECG)
# print('=============EEG=============')
# print(type(data_EEG))
# print(data_EEG.shape)
# # print(data_EEG)

# ====================================

# 计算心电HRV
dir_root = r'D:\pythonPROJ\secondPaperData\ECG'
filelist = os.listdir(dir_root)
numEpochs = 24
for filename in filelist:
    filePath = os.path.join(dir_root, filename)

    data_ecg = scio.loadmat(filePath)['data'].transpose(2, 1, 0)
    print(data_ecg.shape)

    # hrv_arr = np.zeros((numEpochs, 9))  # 9个指标
    hrv_arr = []
    for i in range(numEpochs):
        singelEpoch = -data_ecg[i].flatten()
        print(singelEpoch.shape)
        peaks, info = nk.ecg_peaks(singelEpoch, sampling_rate=250)
        hrv_indices = nk.hrv(peaks, sampling_rate=250)
        spec_columns = hrv_indices[['HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_HF',
                                    'HRV_LF', 'HRV_VLF', 'HRV_TP']].to_numpy()
        print('spec_columns.shape:', spec_columns.shape)
        hrv_arr.append(spec_columns)
    hrv_arr = np.array(hrv_arr).reshape((24, 9))
    print('hrv_arr.shape:', hrv_arr.shape)
    name = filename.split('_')[1]
    np.save(f'../data/ecg_hrv/T_{name}_hrv.npy', hrv_arr)


