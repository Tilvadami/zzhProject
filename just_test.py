# Author: Thel'Vadamee
# Date: 2024/7/23 17:09

import numpy as np
import os

data_root = r'./data/backup'
filelist = os.listdir(data_root)

label_root = r'./fatigue_labels_3'
labellist = os.listdir(label_root)

for filename, labelname in zip(filelist, labellist):
    new_data = []
    new_labels = []
    data = np.load(os.path.join(data_root, filename))
    labels = np.load(os.path.join(label_root, labelname))
    print(data.shape)
    for i in range(data.shape[0]):
        new_data.append(data[i])
        variance = 0.9
        std_dev = np.sqrt(variance)
        random_list = np.random.normal(loc=data[i].mean(), scale=std_dev, size=23)
        for j in range(len(random_list)):
            temp_data = data[i] + random_list[j]
            new_data.append(temp_data)

    new_labels = [i for i in labels for _ in range(24)]
    print(new_labels)
    np.save(f'./changbiaoqian_3/{labelname}', new_labels)

    new_data = np.array(new_data)
    print(new_data.shape)
    # np.save(f'./data/长数据/{filename}', new_data)


