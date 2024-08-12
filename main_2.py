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
import scipy.io as scio
import pandas as pd

# 跨被试策略：18：1

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

batch_size = 64
num_epochs = 300  # 训练轮数
learning_rate = 1e-4
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

combined_data = torch.tensor(whole_de, dtype=torch.float)
combined_label = torch.tensor(whole_labels, dtype=torch.float)

print('combined_data:', combined_data.shape)
print('combined_label:', combined_label.shape)

# 设置记录文件的名字和路径
randomFileName = time.strftime("%Y%m%d-%H%M%S")
filepth111 = os.path.join('./recording', randomFileName)
os.mkdir(filepth111)
with open(os.path.join(filepth111, 'readme.txt'), 'w') as file:
    file.write(f'batch_size: {batch_size}\nlr: {learning_rate}\n'
               f'n_epochs: {num_epochs}\n')

# 整体平均值
total_avg_acc = 0.
# 每一次的模型重新定义
for i in range(combined_data.shape[0]):

    print('トレーニング開始!')
    print('被试：', filelist[i])

    # 分割测试集
    test_data = torch.tensor(combined_data[i], dtype=torch.float)
    test_label = torch.tensor(combined_label[i], dtype=torch.float)

    # 分割训练集
    train_data = torch.cat((combined_data[:i], combined_data[i+1:]), dim=0)
    train_label = torch.cat((combined_label[:i], combined_label[i+1:]), dim=0)

    # 调整训练集的维度
    train_data = train_data.view(-1, train_data.shape[2], train_data.shape[3])
    train_label = train_label.view(-1)

    # 创建DataLoader
    train_dataset = TensorDataset(train_data, train_label)
    test_dataset = TensorDataset(test_data, test_label)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 初始化模型
    model = Model(xdim=[batch_size, channel_num, sample_num], kadj=2, num_out=64, dropout=0.5).to(device)
    # 损失函数
    loss_func = nn.CrossEntropyLoss()  # 交叉熵
    # 优化器
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    # opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    train_len = train_data.shape[0]
    test_len = test_data.shape[0]

    # 存放指标数据
    metrics = []
    for epoch in range(num_epochs):

        # 初始化计数器
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        total_train_acc = 0.
        total_train_loss = 0.

        model.train()
        for de, labels in train_loader:
            de = de.to(device)
            labels = labels.to(device)

            output = model(de)
            train_loss = loss_func(output, labels.long())
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            train_acc = (output.argmax(dim=1) == labels).sum()
            total_train_loss = total_train_loss + train_loss.item()
            total_train_acc += train_acc.item()

        train_loss_list.append(total_train_loss / (len(train_loader)))
        train_acc_list.append(total_train_acc / train_len)

        total_test_acc = 0.
        total_test_loss = 0.

        model.eval()
        with torch.no_grad():
            for de, labels in test_loader:
                de = de.to(device)
                labels = labels.to(device)
                output = model(de)

                predictions = output.argmax(dim=1)

                TP += ((predictions == 1) & (labels == 1)).sum().item()
                TN += ((predictions == 0) & (labels == 0)).sum().item()
                FP += ((predictions == 1) & (labels == 0)).sum().item()
                FN += ((predictions == 0) & (labels == 1)).sum().item()

                test_loss = loss_func(output, labels.long())
                test_acc = (predictions == labels).sum()
                total_test_loss = total_test_loss + test_loss.item()
                total_test_acc += test_acc.item()

        # 这是一轮的指标
        accuracy = (TP + TN) / (TP + TN + FP + FN)      # 准确率
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0      # 查准率(精确率)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0     # 召回率
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0       # F1分数

        # 保存这一轮的指标
        metrics.append({
            "num_epoch": epoch,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
        })

        test_loss_list.append(total_test_loss / (len(test_loader)))
        test_acc_list.append(total_test_acc / test_len)

        # print result
        print("Epoch: {}/{} ".format(epoch + 1, num_epochs),
              "Training Loss: {:.4f} ".format(total_train_loss / len(train_loader)),
              "Training Accuracy: {:.4f} ".format(total_train_acc / train_len),
              "Test Loss: {:.4f} ".format(total_test_loss / len(test_loader)),
              "Test Accuracy: {:.4f}".format(accuracy),
              # "Test Accuracy: {:.4f}".format(total_test_acc / test_len),
              "Test Precision: {:.4f}".format(precision),
              "Test Recall: {:.4f}".format(recall),
              "Test F1_Score: {:.4f}".format(f1_score)
              )

    df = pd.DataFrame(metrics)
    df.to_excel(os.path.join(filepth111, f'index_of_{filelist[i]}.xlsx'), index=False)
    # 创建折线图-train
    # x = [i for i in range(len(train_loss_list))]
    # plt.plot(x, train_loss_list, label='loss')
    # plt.plot(x, train_acc_list, label='acc')

    # 添加标题和标签
    # plt.title('train-loss & acc')
    # plt.xlabel('Epochs')
    # plt.ylabel('Value')
    # plt.legend()

    # 显示图表
    # plt.show()

    # 创建折线图-test
    # x = [i for i in range(len(test_loss_list))]
    # plt.plot(x, test_loss_list, label='loss')
    # plt.plot(x, test_acc_list, label='acc')

    # 添加标题和标签
    # plt.title('test-loss & acc')
    # plt.xlabel('Epochs')
    # plt.ylabel('Value')
    # plt.legend()

    # 显示图表
    # plt.show()

#     avg_acc = sum(test_acc_list) / num_epochs
#     with open(filepth111, 'a') as file:
#         file.write(f'the {filelist[i]} average acc: {avg_acc}\n')
#
#     total_avg_acc = total_avg_acc + avg_acc / len(filelist)     # 19 平均值
#
# with open(filepth111, 'a') as file:
#     file.write(f'the average Acc of all : {total_avg_acc}\nbatch_size: {batch_size}\nlr: {learning_rate}\n'
#                f'n_epochs: {num_epochs}\nk_fold: {k_fold}\n')




