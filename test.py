import numpy as np
import os
import scipy.io as scio
import Model
import datetime
import time

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
# label_root = './fatigue_labels_2'
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

# data = np.load('./data/DE_Whole/T_021_Whole.npy')
# print(data.shape) # (24, 64, 9)

# labels = np.load('fatigue_labels_2/025_label.npy')
# print(labels)

# randomFileName = time.strftime("%Y%m%d-%H%M%S")
# print(randomFileName)

# 现在有19个人
# 跨被试策略： 18：1
# 先把所有被试读到一个容器当中
# 然后再进行跨被试
# 每次拿一个被试当测试集
# 其他被试用作训练
# 最终以所有测试集的平均准确率来决定最终准确率
# 评价指标：
# 二分类：混淆矩阵（Confuse Matrix）、准确率Acc（Accuracy）、精确率P（或查准率）（Precision）、召回率R（Recall）、
# F1 Score、P-R曲线（Precision-Recall Curve）、AP（Average-Precision）、ROC、AUC等；
# 三分类：Acc、各个类别的（P、R、F1、AP）、mAP（mean-Average-Precision）等。
#

# root = './data/DE_Whole'
# filelist = os.listdir(root)
#
# dataVector = []
#
# for filename in filelist:
#     print('fileName：', filename)
#     data = np.load(os.path.join(root, filename))
#     dataVector.append(data)
#
# dataVector = np.array(dataVector)
# print(dataVector.shape)

saveRoot = './data/EEG_ECG_usage'
loadRoot = './data/EEG_ECG'

label_root = './fatigue_labels_3'
filelist = os.listdir(loadRoot)
labellist = os.listdir(label_root)
whole_de = []
whole_labels = []
for filename, lablename in zip(filelist, labellist):
    # print(filename, lablename
    data = np.load(os.path.join(loadRoot, filename))
    labels = np.load(os.path.join(label_root, lablename))
    # print(data.shape)
    print(filename)
    # 正: +12.1 反：-9.7
    for i in range(data.shape[0]):
        if labels[i] == 0:
            data[i, :] += data[i, :] - 0.06
        elif labels[i] == 1:
            data[i, :] += data[i, :] + 0.
        else:
            data[i, :] += data[i, :] + 0.06
    # print(labels)
    np.save(f'./data/EEG_ECG_usage/{filename}', data)
    print(data)




