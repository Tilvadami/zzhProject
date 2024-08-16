# Author: Thel'Vadamee
# Date: 2024/8/15 10:58

import numpy as np
import os

import torch
import Model

# data = np.load('data/EEG_ECG_usage/T_009_Whole.npy')
# print(data.shape)
# print(data)
for _ in range(10):
    dei = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.randn((24, 64, 9)).to(dei)
    mod = Model.Model(xdim=[24, 64, 9], kadj=2, num_out=64, dropout=0.5).to(dei)

    res = mod(data)
    # print(res)
    print(res.argmax(dim=1))
