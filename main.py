import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch import nn
from Index_calculation import testclass
from Model import Model
import os

# batch_size = 256
# num_epochs = 200
# learning_rate = 0.01
# channel_num = 16
# band_num = 5

batch_size = 16
num_epochs = 200    # 训练轮数？
learning_rate = 0.01
channel_num = 64
band_num = 5
sample_num = 9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 将所有的DE读出来放到一个变量里面
root = './data/DE_Whole'
label_root = './fatigue_labels'
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

# 十折交叉验证
kfold = KFold(n_splits=10, shuffle=True)

for train_idx, test_idx in kfold.split(MyDataset):
    print('test...')
    train_data = Subset(MyDataset, train_idx)
    test_data = Subset(MyDataset, test_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # model = Model(xdim=[batch_size, channel_num, band_num], kadj=2, num_out=16, dropout=0.5).to(device)
    model = Model(xdim=[batch_size, channel_num, sample_num], kadj=2, num_out=64, dropout=0.5).to(device)
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    G = testclass()
    train_len = G.len(len(train_idx), batch_size)
    test_len = G.len(len(test_idx), batch_size)

    for epoch in range(num_epochs):
    # -------------------------------------------------
        total_train_acc = 0
        total_train_loss = 0

        for de, labels in train_loader:
            de = de.to(device)
            labels = labels.to(device)

            output = model(de)
#             # print("output:", output.shape)
#
#             train_loss = loss_func(output, labels.long())
#
#             opt.zero_grad()
#             train_loss.backward()
#             opt.step()
#
#             train_acc = (output.argmax(dim=1) == labels).sum()
#
#             train_loss_list.append(train_loss)
#             total_train_loss = total_train_loss + train_loss.item()
#
#             train_acc_list.append(train_acc)
#             total_train_acc += train_acc
#
#         train_loss_list.append(total_train_loss / (len(train_loader)))
#         train_acc_list.append(total_train_acc / train_len)
#
#     # -------------------------------------------------
#         total_test_acc = 0
#         total_test_loss = 0
#
#         with torch.no_grad():
#             for de, labels in test_loader:
#                 de = de.to(device)
#                 labels = labels.to(device)
#
#                 output = model(de)
#                 test_loss = loss_func(output, labels.long())
#
#                 test_acc = (output.argmax(dim=1) == labels).sum()
#
#                 test_loss_list.append(test_loss)
#                 total_test_loss = total_test_loss + test_loss.item()
#
#                 test_acc_list.append(test_acc)
#                 total_test_acc += test_acc
#
#         test_loss_list.append(total_test_loss / (len(test_loader)))
#         test_acc_list.append(total_test_acc / test_len)
#
#         # print result
#         print("Epoch: {}/{} ".format(epoch + 1, num_epochs),
#               "Training Loss: {:.4f} ".format(total_train_loss / len(train_loader)),
#               "Training Accuracy: {:.4f} ".format(total_train_acc / train_len),
#               "Test Loss: {:.4f} ".format(total_test_loss / len(test_loader)),
#               "Test Accuracy: {:.4f}".format(total_test_acc / test_len)
#               )

