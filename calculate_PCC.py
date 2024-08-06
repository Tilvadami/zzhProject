import numpy as np
from sklearn.decomposition import PCA
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 特征向量归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# PCA降维 降到: 通道*DE数量
def sby_dim_re(pcc):
    pca1 = PCA(n_components=1)
    pca2 = PCA(n_components=1)

    pcc_3 = pcc.reshape(pcc.shape[0], -1).transpose(1, 0)  # EEG(926, 560)->(560, 926)
    pca1.fit(pcc_3)
    pcc_3 = pca1.fit_transform(pcc_3)
    # print(pcc_3.shape)

    pcc_2 = pcc_3.reshape(pcc.shape[1], -1).transpose(1, 0)  # EEG(926, 560)->(560, 926)
    pca2.fit(pcc_2)
    pcc_2 = pca2.fit_transform(pcc_2)
    # print(pcc_2.shape)

    pcc_1 = pcc_2.reshape(pcc.shape[2], -1)  # EEG(926, 560)->(560, 926)
    # print(pcc_1.shape)

    return pcc_1


def sby_dim_re4EEG(pcc):
    pca1 = PCA(n_components=1)
    pca2 = PCA(n_components=1)

    # 把第0维降到1
    pcc_3 = pcc.reshape(pcc.shape[0], -1).transpose(1, 0)  # (24, 5*63*300)->(94500 , 24)
    pcc_3 = pca1.fit_transform(pcc_3)

    # # 把第1维降到1
    pcc_2 = pcc_3.reshape(pcc.shape[1], -1).transpose(1, 0)  # EEG(926, 560)->(560, 926
    pcc_2 = pca2.fit_transform(pcc_2)

    # # 把第三位
    pcc_1 = pcc_2.reshape(pcc.shape[2], -1)  # (63, )

    pca = PCA(n_components=9)
    result = pca.fit_transform(pcc_1)
    # print(result.shape)
    return result


def sby_dim_re4ECG(pcc):  # 输入[24, 9]
    pca = PCA(n_components=1)

    ecg_data = pcc.transpose(1, 0)
    result = pca.fit_transform(ecg_data).transpose(1, 0)
    # print(result.shape)
    return result


# 原始laplacian矩阵
def unnormalized_laplacian(adj_matrix):
    # 先求度矩阵
    R = np.sum(adj_matrix, axis=1)
    degreeMatrix = np.diag(R)
    return degreeMatrix - adj_matrix


if __name__ == '__main__':
    DE_list = []
    # 唤醒分数
    ScoreArousal_label_list = []

    ecg_root = './data/ecg_hrv'
    eeg_root = './data/eeg'
    labels_root = './fatigue_labels'

    ecg_hrv_list = os.listdir(ecg_root)
    eeg_de_list = os.listdir(eeg_root)
    labels_list = os.listdir(labels_root)

    for ecgName, eegName, labelName in zip(ecg_hrv_list, eeg_de_list, labels_list):
        print("-=-=-=-: i = ", eegName)
        # print('========', ecg, eeg, label)
        ecg = np.load(os.path.join(ecg_root, ecgName))
        eeg = np.load(os.path.join(eeg_root, eegName))
        # label = np.load(os.path.join(labels_root, labelName))

        eeg_pca = sby_dim_re4EEG(eeg)
        ecg_pca = sby_dim_re4ECG(ecg)

        # 计算皮尔逊相关系数
        rho = np.corrcoef(eeg_pca, ecg_pca)  # (64, 64) ?

        metric_63 = rho[:63, :63]
        metric_1 = rho[63, 63].reshape(1, 1)
        metric_63_1 = rho[:63, 63].reshape(63, 1)

        # print(metric_63_1.shape)

        # 归一化拉普拉斯特征值
        # 脑脑
        nor_lp_63 = unnormalized_laplacian(metric_63)
        lpls_eigenvalue_63, _ = np.linalg.eigh(nor_lp_63)
        lpls_eigenvalue_63 = normalization(lpls_eigenvalue_63)  # 归一化

        # 奇异值分解
        U, _, Vh = np.linalg.svd(metric_63_1)

        # 归一化拉普拉斯特征值
        U_lp = unnormalized_laplacian(U)
        lpls_eigenvalue_63, _ = np.linalg.eigh(U_lp)  # (14,)
        lpls_eigenvalue_63 = normalization(lpls_eigenvalue_63)

        # 再将一次维
        pca = PCA(n_components=9)
        eeg = eeg.reshape(-1, 300)
        eeg = pca.fit_transform(eeg).reshape(24, 5, 63, 9)
        pca_1 = PCA(n_components=1)
        eeg = eeg.reshape(-1, 5)
        eeg = pca_1.fit_transform(eeg).reshape(24, 1, 63, 9)

        # de * weight
        eeg_weight = (eeg.transpose(0, 1, 3, 2) * lpls_eigenvalue_63).reshape(24, 9, 63)  # (926, 5, 8, 14)


        ecg = ecg.reshape((24, 9, 1))

        eeg_ecg = np.concatenate([eeg_weight * lpls_eigenvalue_63, ecg], axis=2).transpose(0, 2, 1)  # (926, 5, 8, 16)
        print(eeg_ecg.shape)
        num = eegName.split('_')[1]
        np.save(f'./data/DE_Whole/T_{num}_Whole.npy', eeg_ecg)


    # for i in range(1, 24):
    #     # 以‘微分熵’作为输入
    #     eeg = np.load(
    #         f"DE_Whole/{i}-DE_Whole-EEG.npy")  # 脑电(926, 5, 14, 8) 我的：(24, 5, 63, 300) 24:epochs 5:bands 63:chs 300:pnts(DE_Whole)
    #     ecg = np.load(f"DE_Whole/{i}-DE_Whole-ECG.npy")  # 心电(926, 5, 2, 8) 我的：(24, 9) ? (24, 5, 1, 300)
    #     # 标签
    #     label = np.load(rf"D:\Program Files\JetBrains\project\zzh_project\label\\{i}-ScoreArousal-label.npy")
    #
    #     # 降维
    #     eeg_pca = sby_dim_re(eeg)  # (14, 8)
    #     ecg_pca = sby_dim_re(ecg)  # (2, 8)
    #
    #     # 计算皮尔逊相关系数
    #     rho = np.corrcoef(eeg_pca, ecg_pca)  # (16, 16) ?
    #     metric_14 = rho[:14, :14]  # (14, 14)
    #     metric_2 = rho[14:17, 14:17]  # (2, 2)
    #     metric_14_2 = rho[:14, 14:17]  # (14, 2)
    #
    #     # 归一化拉普拉斯特征值
    #       # 脑脑
    #     nor_lp_14 = unnormalized_laplacian(metric_14)
    #     lpls_eigenvalue_14, _ = np.linalg.eigh(nor_lp_14)  # (14,)
    #     lpls_eigenvalue_14 = normalization(lpls_eigenvalue_14)
    #       心心
    #     nor_lp_2 = unnormalized_laplacian(metric_2)
    #     lpls_eigenvalue_2, _ = np.linalg.eigh(nor_lp_2)  # (2,)
    #     lpls_eigenvalue_2 = normalization(lpls_eigenvalue_2)
    #
    #     # 奇异值分解   脑心
    #     U, _, Vh = np.linalg.svd(metric_14_2)  # (14, 14), (2, 2)
    #
    #     # 归一化拉普拉斯特征值
    #     U_lp = unnormalized_laplacian(U)
    #     lpls_U_eigVal_14, _ = np.linalg.eigh(U_lp)  # (14,)
    #     lpls_U_eigVal_14 = normalization(lpls_U_eigVal_14)
    #
    #     # 归一化拉普拉斯特征值
    #     Vh_lp = unnormalized_laplacian(Vh)
    #     lpls_Vh_eigVal_2, _ = np.linalg.eigh(Vh_lp)  # (2,)
    #     lpls_Vh_eigVal_2 = normalization(lpls_Vh_eigVal_2)
    #
    #     # de * weight
    #     eeg_weight = eeg.transpose(0, 1, 3, 2) * lpls_eigenvalue_14  # (926, 5, 8, 14)
    #     ecg_weight = ecg.transpose(0, 1, 3, 2) * lpls_eigenvalue_2  # (926, 5, 8, 2)
    #
    #     # concatenate channel
    #     eeg_ecg = np.concatenate([eeg_weight * lpls_U_eigVal_14, ecg_weight * lpls_Vh_eigVal_2],
    #                              axis=3)  # (926, 5, 8, 16)
    #     eeg_ecg = eeg_ecg.transpose(0, 3, 1, 2)  # (926, 16, 5, 8)
    #     print(eeg_ecg.shape)
    #     np.save(rf"D:\Program Files\JetBrains\project\zzh_project\weight_de\DE_Whole{i}.npy", eeg_ecg)
    #     DE_list.append(eeg_ecg)
    #     ScoreArousal_label_list.append(label)
    #
    # DE_Whole = np.concatenate(DE_list)
    # labels = np.concatenate(ScoreArousal_label_list)
    # print(DE_Whole.shape)
    # print(labels.shape)
    # # np.save("D:\Program Files\JetBrains\project\zzh_project\DE_Whole.npy", DE_Whole)
    # np.save("D:\Program Files\JetBrains\project\zzh_project\ScoreArousal_label.npy", labels)
