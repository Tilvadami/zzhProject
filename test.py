import numpy as np
import os
import scipy.io as scio


# a = np.load(rf"D:\Program Files\JetBrains\project\zzh_project\label\\1-ScoreArousal-label.npy")
# b = np.load(rf"D:\Program Files\JetBrains\project\zzh_project\label\\1-ScoreDominance-label.npy")
# c = np.unique(a)
# print(c)

# root = r'D:\pythonPROJ\secondPaperData\EEG\T_022_REST_epochs.mat'
# datamat = scio.loadmat(root)
#
# data = datamat['EEG']
# print(data.shape)


filelist = os.listdir('./data/DE_Whole')
for filename in filelist:
    filePath = os.path.join('./data/DE_Whole', filename)
    data = np.load(filePath)
    print(data)


