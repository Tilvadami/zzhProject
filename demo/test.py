# Author: Thel'Vadamee
# Date: 2024/7/29 10:25

import numpy as np
import os

data_root = r'../DE'

filelist = os.listdir(data_root)
print(filelist)

demo_pth_ECG = r'../DE/1-DE-ECG.npy'
demo_pth_EEG = r'../DE/1-DE-EEG.npy'
data_ECG = np.load(demo_pth_ECG)
data_EEG = np.load(demo_pth_EEG)
print('=============ECG=============')
print(type(data_ECG))
print(data_ECG.shape)
# print(data_ECG)
print('=============EEG=============')
print(type(data_EEG))
print(data_EEG.shape)
# print(data_EEG)

