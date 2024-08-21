# Author: Thel'Vadamee
# Date: 2024/8/20 17:18
import pandas as pd
import os

root = r'recording/20240820-202539'

filelist = os.listdir(root)
max_value_list = []
for i in range(len(filelist)):
    # 获得后缀名
    filename = filelist[i]
    extension = os.path.splitext(filename)[1]
    if extension == '.txt':
        continue

    filePath = os.path.join(root, filename)
    df = pd.read_excel(filePath)
    v = max(df['Accuracy'])
    print(f'{i}======{v}')
    max_value_list.append(v)

print(len(max_value_list))
print('平均值:', sum(max_value_list)/len(max_value_list))
