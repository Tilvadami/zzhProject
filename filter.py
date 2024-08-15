import math
import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter
import os
import scipy.io as scio

'''
file name: DE_3D_Feature
input: the path of saw EEG file in SEED-VIG dataset
output: the 3D feature of all subjects
'''


# step1: input raw data
# step2: decompose frequency bands
# step3: calculate DE_Whole
# step4: stack them into 3D featrue


def butter_bandpass(low_freq, high_freq, fs, order=5):
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# calculate DE_Whole
def calculate_DE(saw_EEG_signal):
    variance = np.var(saw_EEG_signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


# filter for 5 frequency bands
def butter_bandpass_filter(data: object, low_freq: object, high_freq: object, fs: object, order: object = 5) -> object:
    b, a = butter_bandpass(low_freq, high_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


def decompose_to_DE(EEGDATA):
    # EEGDATA:[75000,63]
    # 这里我打算把24个段分别拿来计算微分熵
    # sampling rate 采样率
    frequency = 250

    # samples 75000
    samples = EEGDATA.shape[0]

    # 100 samples = 1 DE_Whole
    # 1s 一个de
    num_sample = int(samples / frequency)
    channels = EEGDATA.shape[1]
    bands = 5
    # init DE_Whole [75000, 17, 5]
    # DE_3D_feature = np.empty([num_sample, channels, bands]) [300, 63, 5]

    # feature = np.empty([samples, channels, bands])    [75000, 63, 5]


    fea = []
    for channel in range(channels):
        feature = np.empty([0, samples])  # [0, 75000]  空数组没有指定行数
        temp_de = np.empty([0, num_sample])  # [0, 300]空数组

        trial_signal = EEGDATA[:, channel]
        # get 5 frequency bands
        delta = butter_bandpass_filter(trial_signal, 1, 4, frequency, order=3)
        theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
        alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
        beta = butter_bandpass_filter(trial_signal, 14, 30, frequency, order=3)
        gamma = butter_bandpass_filter(trial_signal, 30, 45, frequency, order=3)

        # feature = np.stack((delta, theta, alpha, beta, gamma), axis=0)  # 应该是(5, 75000)
        # print('feature.shape', feature.shape)
        # fea.append(feature)  # 列表    最终会形成[63, 5, 75000]

        DE_delta = np.zeros(shape=[0], dtype=float)  # [0]表示这个数组是一维的，并且元素为0
        DE_theta = np.zeros(shape=[0], dtype=float)
        DE_alpha = np.zeros(shape=[0], dtype=float)
        DE_beta = np.zeros(shape=[0], dtype=float)
        DE_gamma = np.zeros(shape=[0], dtype=float)
        # DE_Whole of delta, theta, alpha, beta and gamma
        # 这里的num_sample代表要计算多少个DE值, 这里是75000/250 = 300
        for index in range(num_sample):  # index从0开始
            DE_delta = np.append(DE_delta, calculate_DE(delta[index * frequency:(index + 1) * frequency]))
            DE_theta = np.append(DE_theta, calculate_DE(theta[index * frequency:(index + 1) * frequency]))
            DE_alpha = np.append(DE_alpha, calculate_DE(alpha[index * frequency:(index + 1) * frequency]))
            DE_beta = np.append(DE_beta, calculate_DE(beta[index * frequency:(index + 1) * frequency]))
            DE_gamma = np.append(DE_gamma, calculate_DE(gamma[index * frequency:(index + 1) * frequency]))

        temp_de = np.vstack([temp_de, DE_delta])
        temp_de = np.vstack([temp_de, DE_theta])
        temp_de = np.vstack([temp_de, DE_alpha])
        temp_de = np.vstack([temp_de, DE_beta])
        temp_de = np.vstack([temp_de, DE_gamma])

        fea.append(temp_de)

    fea = np.array(fea)
    print('-=-=-=-=-=', fea.shape)
    fea = fea.transpose([1, 0, 2])
    return fea


if __name__ == '__main__':
    # 原始数据
    # filePath = r"D:\BaiduNetdiskDownload\Emotion_SEED-IV\1\\"
    # dataName = ['1_20160518.mat.npy', '2_20150915.mat.npy', '3_20150919.mat.npy', '4_20151111.mat.npy', '5_20160406.mat.npy',
    #             '6_20150507.mat.npy', '7_20150715.mat.npy', '8_20151103.mat.npy', '9_20151028.mat.npy', '10_20151014.mat.npy',
    #             '11_20150916.mat.npy', '12_20150725.mat.npy', '13_20151115.mat.npy', '14_20151205.mat.npy', '15_20150508.mat.npy']
    # dataName = '1_20151124_noon_2.mat'

    # X = np.empty([0, 27, 5])

    # 原始数据
    rootPath = r'D:\pythonPROJ\secondPaperData\EEG-rest'
    # rootPath = r'D:\pythonPROJ\secondPaperData\ECG'
    # filelist = os.listdir(rootPath)
    filelist = ['T_001_REST_epochs.mat']
    for filename in filelist:
        print(filename)
        filePath = os.path.join(rootPath, filename)
        dataMat = scio.loadmat(filePath)
        print(dataMat.keys())

        dataMat['EEG'] = dataMat['EEG']['data'][0][0]
        print(dataMat['EEG'].shape)
        EEGData = dataMat['EEG'].transpose(2, 1, 0)
        print(EEGData.shape)

        segments = EEGData.shape[0]
        EEG_DE = []
        for seg in range(segments):
            trail_data = EEGData[seg]
            print(trail_data.shape)
            EEG_DE.append(decompose_to_DE(trail_data))

        EEG_DE = np.array(EEG_DE)
        print('处理后的：', EEG_DE.shape)

        num_name = filename.split('_')[1]
        np.save(f'./data/eeg/T_{num_name}_DE.npy', EEG_DE)

    # for i in range(len(dataName)):
    #     dataFile = filePath + dataName[i]
    #     print('processing {}'.format(dataName[i]))
    #     # print(dataFile)
    #     # every subject DE_Whole feature
    #     DE_3D_feature = decompose_to_DE(dataFile)
    #     # name = dataName[i].split(".")[0]
    #     path = r"E:\ZHK\SEED-IV\six-area\1-test_other-train\filter_raw_data\\" + dataName[i].split('.')[0] + '.npy'
    #
    #     # print(path)
    #     np.save(path, DE_3D_feature)
    #     # # all subjects
    #     # X = np.vstack([X, DE_3D_feature])

# dataFile = filePath + dataName
# DE_3D_feature = decompose_to_DE(dataFile)
# X = np.vstack([X, DE_3D_feature])

# save .npy file
# np.save("D:/BaiduNetdiskDownload/SEED-VIG/Raw_Data/processedData/test1_data.npy", X)
