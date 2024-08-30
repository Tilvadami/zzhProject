# Author: Thel'Vadamee
# Date: 2024/8/15 10:58

import numpy as np
import os

import torch
import Model

import pandas as pd
import scipy.io as scio

# data = np.load('data/EEG_ECG_usage/T_009_Whole.npy')
# print(data.shape)
# print(data)
# for _ in range(10):
#     dei = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     data = torch.randn((24, 64, 9)).to(dei)
#     mod = Model.Model(xdim=[24, 64, 9], kadj=2, num_out=64, dropout=0.5).to(dei)
#
#     res = mod(data)
#     # print(res)
#     print(res.argmax(dim=1))

# root = r'./fatigue_labels_2'
# filelist = os.listdir(root)
# # filelist = ['021_label.npy']
# for filename in filelist:
#     print(filename)
#     data = np.load(os.path.join(root, filename))
#     print(data)


# root = r'D:\2023疲劳KJW数据\15导的数据（心电就不提取了，用之前的）'
# filelist = os.listdir(root)
#
# for filename in filelist:
#     filePath = os.path.join(root, filename)
#     data = scio.loadmat(filePath)
#
#     print(data['data'].shape)

# root = r'data/EEG_ECG_usage'
# matSaveRoot = r'data/17通道(16EEG+ECG)Data/mat'
# npySaveRoot = r'data/17通道(16EEG+ECG)Data/npy'
#
# filelist = os.listdir(root)
# for filename in filelist:
#     filePath = os.path.join(root, filename)
#     data = np.load(filePath)
#     print(data.shape)
#     selected_data = data[:, -17:, :]
#     print("selected_data.shape:", selected_data.shape)
#     # print(selected_data)
#
#     matData = {'data': selected_data}
#     lst = filename.split('.')
#     print(lst[0])
#     matSavePath = os.path.join(matSaveRoot, lst[0]+'.mat')
#     npySavePath = os.path.join(npySaveRoot, filename)
#     # scio.savemat(matSavePath, matData)
#     np.save(npySavePath, selected_data)

channelsToSelect = ['Fp1', 'Fp2', 'F4', 'F7', 'F3', 'F8', 'C3', 'C4', 'P4', 'P5', 'P6', 'P3', 'O2', 'O1', 'FC6', 'FC5']

root = r'D:\2023疲劳KJW数据\17导的数据'
saveRoot = r'D:\2023疲劳KJW数据\17导的数据\yes'

filelist = os.listdir(root,)
for filename in filelist:
    filePath = os.path.join(root, filename)
    # 判断是否为文件夹
    if os.path.isdir(filePath):
        continue
    dataMat = scio.loadmat(filePath)
    eegData = dataMat['EEG']

    EEG_dict = {}
    for i in range(len(channelsToSelect)):
        ch = channelsToSelect[i]
        EEG_dict[ch] = eegData[i]

    new_mat = {'ECG': dataMat['ECG'], 'EEG': EEG_dict}
    saveMatPath = os.path.join(saveRoot, filename)
    # 保存
    scio.savemat(saveMatPath, new_mat)



