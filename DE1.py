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
def butter_bandpass_filter(data, low_freq, high_freq, fs, order=5):
    b, a = butter_bandpass(low_freq, high_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


def decompose_to_DE(file):
    # read data  sample * channel [1416000, 17]
    data = loadmat(file)['EEG']['data'][0, 0]
    # sampling rate
    frequency = 200
    # samples 1416000
    samples = data.shape[0]
    # 100 samples = 1 DE_Whole
    num_sample = int(samples / 100)
    channels = data.shape[1]
    bands = 5
    # init DE_Whole [141600, 17, 5]
    # DE_3D_feature = np.empty([num_sample, channels, bands])

    # feature = np.empty([samples, channels, bands])

    # temp_de = np.empty([0, num_sample])

    # feature = np.empty([0, samples])
    fea = []
    for channel in range(channels):
        trial_signal = data[:, channel]
        # get 5 frequency bands
        delta = butter_bandpass_filter(trial_signal, 1, 4, frequency, order=3)
        theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
        alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
        beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
        gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

        # print(np.array(delta).shape)

        feature = np.stack((delta, theta, alpha, beta, gamma), axis=0)
        fea.append(feature)     # 一个通道一个通道地append
        # feature = np.stack([feature, delta])
        # feature = np.stack([feature, theta])
        # feature = np.stack([feature, alpha])
        # feature = np.stack([feature, beta])
        # feature = np.stack([feature, gamma])

        # temp = np.vstack([temp, delta])
        # temp = np.vstack([temp, theta])
        # temp = np.vstack([temp, alpha])
        # temp = np.vstack([temp, beta])
        # temp = np.vstack([temp, gamma])

        ########################################################################################
        # DE_Whole
        # DE_delta = np.zeros(shape=[0], dtype=float)
        # DE_theta = np.zeros(shape=[0], dtype=float)
        # DE_alpha = np.zeros(shape=[0], dtype=float)
        # DE_beta = np.zeros(shape=[0], dtype=float)
        # DE_gamma = np.zeros(shape=[0], dtype=float)
        # # DE_Whole of delta, theta, alpha, beta and gamma
        # for index in range(num_sample):
        #     DE_delta = np.append(DE_delta, calculate_DE(delta[index * 100:(index + 1) * 100]))
        #     DE_theta = np.append(DE_theta, calculate_DE(theta[index * 100:(index + 1) * 100]))
        #     DE_alpha = np.append(DE_alpha, calculate_DE(alpha[index * 100:(index + 1) * 100]))
        #     DE_beta = np.append(DE_beta, calculate_DE(beta[index * 100:(index + 1) * 100]))
        #     DE_gamma = np.append(DE_gamma, calculate_DE(gamma[index * 100:(index + 1) * 100]))
        #
        # temp_de = np.vstack([temp_de, DE_delta])
        # temp_de = np.vstack([temp_de, DE_theta])
        # temp_de = np.vstack([temp_de, DE_alpha])
        # temp_de = np.vstack([temp_de, DE_beta])
        # temp_de = np.vstack([temp_de, DE_gamma])

    ########################################################################################

    # temp_trial_de = temp_de.reshape(-1, 5, num_sample)
    # temp_trial_de = temp_trial_de.transpose([2, 0, 1])
    # DE_3D_feature = np.vstack([temp_trial_de])
    # feature_trial = feature.reshape(17, 5, -1)
    feature_trial = np.stack(fea, axis=0)
    # print(feature.shape)
    feature_trial = feature_trial.transpose([2, 0, 1])
    feature = feature_trial.reshape(885, 1600, 17, 5).transpose([0, 3, 2, 1])
    print(feature.shape)

    # return DE_3D_feature
    return feature


if __name__ == '__main__':
    # Fill in your SEED-VIG dataset path
    filePath = r"D:\BaiduNetdiskDownload\SEED-VIG\Raw_Data\processedData\brain area\raw_data\\"
    dataName = ['1_20151124_noon_2.mat', '2_20151106_noon.mat', '3_20151024_noon.mat', '4_20151105_noon.mat',
                '4_20151107_noon.mat', '5_20141108_noon.mat', '5_20151012_night.mat', '6_20151121_noon.mat',
                '7_20151015_night.mat', '8_20151022_noon.mat', '9_20151017_night.mat', '10_20151125_noon.mat',
                '11_20151024_night.mat', '12_20150928_noon.mat', '13_20150929_noon.mat', '14_20151014_night.mat',
                '15_20151126_night.mat', '16_20151128_night.mat', '17_20150925_noon.mat', '18_20150926_noon.mat',
                '19_20151114_noon.mat', '20_20151129_night.mat', '21_20151016_noon.mat']
    # dataName = '1_20151124_noon_2.mat'

    X = np.empty([0, 17, 5])

    for i in range(len(dataName)):
        dataFile = filePath + dataName[i]
        print('processing {}'.format(dataName[i]))
        # every subject DE_Whole feature
        DE_3D_feature = decompose_to_DE(dataFile)
        name = dataName[i].split(".")[0]
        path = r"D:\BaiduNetdiskDownload\SEED-VIG\Raw_Data\processedData\brain area\split" + '\\' + name + ".npy"
        # print(path)
        np.save(path, DE_3D_feature)
        # # all subjects
        # X = np.vstack([X, DE_3D_feature])

    # dataFile = filePath + dataName
    # DE_3D_feature = decompose_to_DE(dataFile)
    # X = np.vstack([X, DE_3D_feature])

    # save .npy file
    # np.save("D:/BaiduNetdiskDownload/SEED-VIG/Raw_Data/processedData/test1_data.npy", X)
