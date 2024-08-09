# Author: Thel'Vadamee
# Date: 2024/7/23 17:09

import numpy as np
import os

filelist = os.listdir('fatigue_labels_2/')

# print(filelist)

for filename in filelist:
    print(filename)
    labl = np.load('fatigue_labels_2/' + filename)
    print(len(labl))
