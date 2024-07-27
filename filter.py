import math
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter

'''
file name: DE_3D_Feature
input: the path of saw EEG file in SEED-VIG dataset
output: the 3D feature of all subjects
'''

# step1: input raw data
# step2: decompose frequency bands
# step3: calculate DE
# step4: stack them into 3D featrue


def butter_bandpass(low_freq, high_freq, fs, order=5):
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# calculate DE
def calculate_DE(saw_EEG_signal):
    variance = np.var(saw_EEG_signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


# filter for 5 frequency bands
def butter_bandpass_filter(data, low_freq, high_freq, fs, order=5):
    b, a = butter_bandpass(low_freq, high_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


def decompose_to_DE(file):
    # read data  sample * channel [1416000, 17]
    data = np.load(file)
    # print(data.shape)
    data = data[:680800, :]
    # sampling rate
    frequency = 800
    # samples 1416000
    samples = data.shape[0]
    # 100 samples = 1 DE
    # num_sample = int(samples/200)
    channels = data.shape[1]
    bands = 5
    # init DE [141600, 17, 5]
    # DE_3D_feature = np.empty([num_sample, channels, bands])

    # feature = np.empty([samples, channels, bands])

    # temp_de = np.empty([0, num_sample])

    # feature = np.empty([0, samples])
    fea = []
    for channel in range(channels):
        trial_signal = data[:, channel]
        # get 5 frequency bands
        delta = butter_bandpass_filter(trial_signal, 1, 4,   frequency, order=3)
        theta = butter_bandpass_filter(trial_signal, 4, 8,   frequency, order=3)
        alpha = butter_bandpass_filter(trial_signal, 8, 14,  frequency, order=3)
        beta  = butter_bandpass_filter(trial_signal, 14, 30, frequency, order=3)
        gamma = butter_bandpass_filter(trial_signal, 30, 48, frequency, order=3)

        # print(delta.shape)
        # print(theta.shape)
        # print(alpha.shape)
        # print(beta.shape)
        # print(gamma.shape)

        feature = np.stack((delta, theta, alpha, beta, gamma), axis = 0)
        fea.append(feature)

    feature_trial = np.stack(fea, axis=0)
    # print(feature_trial.shape)
    feature_trial = feature_trial.transpose([2, 0, 1])
    feature = feature_trial.reshape(-1, 800, channels, 5).transpose([0, 3, 2, 1])
    print(feature.shape)

    return feature


if __name__ == '__main__':
    # Fill in your SEED-VIG dataset path
    filePath = r"D:\BaiduNetdiskDownload\Emotion_SEED-IV\1\\"
    dataName = ['1_20160518.mat.npy', '2_20150915.mat.npy', '3_20150919.mat.npy', '4_20151111.mat.npy', '5_20160406.mat.npy',
                '6_20150507.mat.npy', '7_20150715.mat.npy', '8_20151103.mat.npy', '9_20151028.mat.npy', '10_20151014.mat.npy',
                '11_20150916.mat.npy', '12_20150725.mat.npy', '13_20151115.mat.npy', '14_20151205.mat.npy', '15_20150508.mat.npy']
    # dataName = '1_20151124_noon_2.mat'

    # X = np.empty([0, 27, 5])

    for i in range(len(dataName)):
        dataFile = filePath + dataName[i]
        print('processing {}'.format(dataName[i]))
        # print(dataFile)
        # every subject DE feature
        DE_3D_feature = decompose_to_DE(dataFile)
        # name = dataName[i].split(".")[0]
        path = r"E:\ZHK\SEED-IV\six-area\1-test_other-train\filter_raw_data\\" + dataName[i].split('.')[0] + '.npy'

        # print(path)
        np.save(path, DE_3D_feature)
        # # all subjects
        # X = np.vstack([X, DE_3D_feature])

    # dataFile = filePath + dataName
    # DE_3D_feature = decompose_to_DE(dataFile)
    # X = np.vstack([X, DE_3D_feature])

    # save .npy file
    # np.save("D:/BaiduNetdiskDownload/SEED-VIG/Raw_Data/processedData/test1_data.npy", X)