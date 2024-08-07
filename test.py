import numpy as np
import os
import scipy.io as scio
import Model


# a = np.load(rf"D:\Program Files\JetBrains\project\zzh_project\label\\1-ScoreArousal-label.npy")
# b = np.load(rf"D:\Program Files\JetBrains\project\zzh_project\label\\1-ScoreDominance-label.npy")
# c = np.unique(a)
# print(c)

# root = r'D:\pythonPROJ\secondPaperData\EEG\T_022_REST_epochs.mat'
# datamat = scio.loadmat(root)
#
# data = datamat['EEG']
# print(data.shape)

# test_data = np.random.randn(128, 16, 5, 8)
#
# print(test_data.shape)
#
# mod = Model.Model(xdim=1, kadj=2, num_out=2, dropout=0.5)

# root = './data/DE_Whole'
# label_root = './fatigue_labels'
# filelist = os.listdir(root)
# labellist = os.listdir(label_root)
# whole_de = []
# whole_labels = []
# for filename, lablename in zip(filelist, labellist):
#     filePath = os.path.join(root, filename)
#     print(filePath)
#     data = np.load(filePath)
#     whole_de.append(data)
#
#     labelPath = os.path.join(label_root, lablename)
#     print(labelPath)
#     label = np.load(labelPath)
#     whole_labels.append(label)
#
# combined_data = np.concatenate(whole_de, axis=0)
# combined_label = np.concatenate(whole_labels, axis=0)
# print(combined_data.shape)
# print(combined_label.shape)

data = np.load('ScoreArousal_label.npy')
print(data.shape)

