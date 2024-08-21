# Author: Thel'Vadamee
# Date: 2024/7/15 11:08

import pandas as pd
import numpy as np


# 1-9: 1 2 3 4 5 6 7 8 9; 1-5:0 6-9:1
def gen_classes_2(score):
    if score >= 1 and score <= 6:
        return 0
    else:
        return 1


# 1-9: 1 2 3 4 5 6 7 8 9; 1-4:0 5-7:1 8-9:2
def gen_classes_3(score):
    if score >= 1 and score <= 4:
        return 0
    elif score >= 5 and score <= 7:
        return 1
    else:
        return 2


# 1-9: 1 2 3 4 5 6 7 8 9; 1-2:0 3-4：1 5-7:2 8-9:3
def gen_classes_4(score):
    if score >= 1 and score <= 2:
        return 0
    elif score >= 3 and score <= 4:
        return 1
    elif score >= 5 and score <= 7:
        return 2
    else:
        return 3


if __name__ == '__main__':

    # excel_root = r'D:\疲劳刻度T001-T020.xlsx'
    excel_root = r'D:\疲劳程度.xlsx'
    df = pd.read_excel(excel_root, header=None)
    flag = '021'
    labels = []

    print('df.szie():', df.size)
    for index, row in df.iterrows():

        Num = row[0]
        score = row[1]
        print(f'{row[0]}----{row[1]}')
        split_list = Num.split('_')
        print(split_list[1])
        # 换被试
        if flag != split_list[1]:
            print('labels.size:', len(labels))
            labels = np.array(labels)
            np.save(f'./fatigue_labels_4/{flag}_label.npy', labels)
            labels = []
            flag = split_list[1]

        labels.append(gen_classes_3(score))
        print(index)
        if index == 479:
            labels = np.array(labels)
            np.save(f'./fatigue_labels_4/{flag}_label.npy', labels)
            break
