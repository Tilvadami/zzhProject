import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch import nn
from Index_calculation import testclass
from Model import Model
import os
import matplotlib.pyplot as plt
import random
import time


def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 固定随机种子，实现可复现性
get_random_seed(43)
# batch_size = 256
# num_epochs = 200
# learning_rate = 0.01
# channel_num = 16
# band_num = 5

batch_size = 16
num_epochs = 300  # 训练轮数
learning_rate = 1e-3    # 0.001-0.1
channel_num = 64
band_num = 5
sample_num = 9
k_fold = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 将所有的DE读出来放到一个变量里面
root = './data/DE_Whole'
label_root = './fatigue_labels_2'
filelist = os.listdir(root)
labellist = os.listdir(label_root)
whole_de = []
whole_labels = []
for filename, lablename in zip(filelist, labellist):
    filePath = os.path.join(root, filename)
    data = np.load(filePath)
    whole_de.append(data)

    labelPath = os.path.join(label_root, lablename)
    label = np.load(labelPath)
    whole_labels.append(label)

combined_data = np.concatenate(whole_de, axis=0)
combined_label = np.concatenate(whole_labels, axis=0)

DE = torch.tensor(combined_data.real.astype(float), dtype=torch.float)
labels = torch.tensor(combined_label, dtype=torch.float)

# DE = torch.tensor(np.load("D:/Program Files/JetBrains/project/zzh_project/DE_Whole.npy").real.astype(float), dtype=torch.float)  # (21298, 16, 5, 8)
# labels = torch.tensor(np.load("D:/Program Files/JetBrains/project/zzh_project/ScoreArousal_label.npy"), dtype=torch.float).squeeze_(1)    # (21298)
# print(eigenvalues.shape)
# print(eigenvector.shape)
# print(labels.shape)

MyDataset = TensorDataset(DE, labels)

while batch_size <= 64:
    learning_rate = 1e-3  # 0.001-0.1
    while learning_rate <= 0.1:
        randomFileName = time.strftime("%Y%m%d-%H%M%S") + ".txt"
        filepth111 = os.path.join('./recording', randomFileName)
        # 整体平均值
        total_avg_acc = 0.

        # 十折交叉验证
        kfold = KFold(n_splits=k_fold, shuffle=True)
        n_fold = 0
        for train_idx, test_idx in kfold.split(MyDataset):
            n_fold = n_fold + 1
            print('トレーニング開始!')
            train_data = Subset(MyDataset, train_idx)
            test_data = Subset(MyDataset, test_idx)

            train_loader = DataLoader(train_data, batch_size=batch_size)
            test_loader = DataLoader(test_data, batch_size=batch_size)

            # 每一折的模型重新初始化
            # model = Model(xdim=[batch_size, channel_num, band_num], kadj=2, num_out=16, dropout=0.5).to(device)
            model = Model(xdim=[batch_size, channel_num, sample_num], kadj=2, num_out=64, dropout=0.5).to(device)
            # 损失函数
            loss_func = nn.CrossEntropyLoss()  # 交叉熵
            # 优化器
            opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)

            train_acc_list = []
            train_loss_list = []
            test_acc_list = []
            test_loss_list = []

            G = testclass()
            # train_len = G.len(len(train_idx), batch_size)
            train_len = len(train_idx)
            # test_len = G.len(len(test_idx), batch_size)
            test_len = len(test_idx)
            model.train()
            for epoch in range(num_epochs):
                # -------------------------------------------------
                total_train_acc = 0.
                total_train_loss = 0.

                for de, labels in train_loader:
                    de = de.to(device)
                    labels = labels.to(device)

                    output = model(de)
                    # print("output:", output.shape)

                    train_loss = loss_func(output, labels.long())
                    # total_train_loss += train_loss.item()
                    opt.zero_grad()
                    train_loss.backward()
                    opt.step()
                    train_acc = (output.argmax(dim=1) == labels).sum()
                    # print('-=-=Acc Num:', train_acc)

                    # train_loss_list.append(train_loss.item())
                    total_train_loss = total_train_loss + train_loss.item()

                    # train_acc_list.append(train_acc.item())
                    total_train_acc += train_acc.item()

                train_loss_list.append(total_train_loss / (len(train_loader)))
                train_acc_list.append(total_train_acc / train_len)

                # -------------------------------------------------
                total_test_acc = 0
                total_test_loss = 0
                model.eval()
                with torch.no_grad():
                    for de, labels in test_loader:
                        de = de.to(device)
                        labels = labels.to(device)

                        output = model(de)
                        test_loss = loss_func(output, labels.long())

                        test_acc = (output.argmax(dim=1) == labels).sum()
                        # print('-=-=-Test Acc Num :', test_acc)

                        # test_loss_list.append(test_loss)
                        total_test_loss = total_test_loss + test_loss.item()

                        # test_acc_list.append(test_acc)
                        total_test_acc += test_acc.item()

                test_loss_list.append(total_test_loss / (len(test_loader)))
                test_acc_list.append(total_test_acc / test_len)

                # print result
                print("Epoch: {}/{} ".format(epoch + 1, num_epochs),
                      "Training Loss: {:.4f} ".format(total_train_loss / len(train_loader)),
                      "Training Accuracy: {:.4f} ".format(total_train_acc / train_len),
                      "Test Loss: {:.4f} ".format(total_test_loss / len(test_loader)),
                      "Test Accuracy: {:.4f}".format(total_test_acc / test_len)
                      )

            # 创建折线图-train
            x = [i for i in range(len(train_loss_list))]
            plt.plot(x, train_loss_list, label='loss')
            plt.plot(x, train_acc_list, label='acc')

            # 添加标题和标签
            plt.title('train-loss & acc')
            plt.xlabel('Epochs')
            plt.ylabel('Value')
            plt.legend()

            # 显示图表
            # plt.show()

            # 创建折线图-test
            x = [i for i in range(len(test_loss_list))]
            plt.plot(x, test_loss_list, label='loss')
            plt.plot(x, test_acc_list, label='acc')

            # 添加标题和标签
            plt.title('test-loss & acc')
            plt.xlabel('Epochs')
            plt.ylabel('Value')
            plt.legend()

            # 显示图表
            # plt.show()

            avg_acc = sum(test_acc_list) / num_epochs
            with open(filepth111, 'a') as file:
                file.write(f'the {n_fold} fold average acc: {avg_acc}\n')

            total_avg_acc = total_avg_acc + avg_acc / 10

        with open(filepth111, 'a') as file:
            file.write(f'the average Acc of all : {total_avg_acc}\nbatch_size: {batch_size}\nlr: {learning_rate}\n'
                       f'n_epochs: {num_epochs}\nk_fold: {k_fold}\n')

        learning_rate = learning_rate + 0.001

    batch_size = batch_size + 1
