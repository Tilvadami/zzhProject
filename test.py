import numpy as np


a = np.load(rf"D:\Program Files\JetBrains\project\zzh_project\label\\1-ScoreArousal-label.npy")
b = np.load(rf"D:\Program Files\JetBrains\project\zzh_project\label\\1-ScoreDominance-label.npy")
c = np.unique(a)
print(c)