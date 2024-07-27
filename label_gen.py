# Author: Thel'Vadamee
# Date: 2024/7/15 11:08

import pandas as pd
import numpy as np


# 1-9: 1 2 3 4 5 6 7 8 9; 1-5:0 6-9:1
def gen_classes_2(score):
    if score >= 1 and score <= 5:
        return 0
    else:
        return 1


# 1-9: 1 2 3 4 5 6 7 8 9; 1-3:0 4-6:1 7-9:2
def gen_classes_3(score):
    if score >= 1 and score <= 3:
        return 0
    elif score >= 4 and score <= 6:
        return 1
    else:
        return 2


if __name__ == '__main__':

    excel_root = r'D:\疲劳程度.xlsx'
    df = pd.read_excel(excel_root, header=None)
    flag = '021'
    labels = []

    print('df.szie():', df.size)
    for index, row in df.iterrows():

        Num = row[0]
        score = row[1]
        # print(f'{row[0]}----{row[1]}')
        split_list = Num.split('_')
        print(split_list[1])
        # 换被试
        if flag != split_list[1]:
            print('labels.size:', len(labels))
            labels = np.array(labels)
            np.save(f'./fatigue_labels/{flag}_label.npy', labels)
            labels = []
            flag = split_list[1]

        labels.append(gen_classes_2(score))
        print(index)
        if index == 479:
            labels = np.array(labels)
            np.save(f'./fatigue_labels/{flag}_label.npy', labels)
            break
