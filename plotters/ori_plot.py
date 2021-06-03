import matplotlib.pyplot as plt
import csv
import numpy as np

ori_gt = []
ori_reg = []
with open("3Dtrain_gt.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for col in csv_reader:
        ori_gt.append(col[2])
ori_gt = np.asarray(ori_gt)
ori_gt = ori_gt.astype(float)
with open("3Dtrain_regout.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for col in csv_reader:
        ori_reg.append(col[2])
ori_reg = np.asarray(ori_reg)
ori_reg = ori_reg.astype(float)
n_samples = np.arange(0, len(ori_reg), 1)
plt.scatter(ori_gt, n_samples, label="g_t", s=10, color="blue", alpha=0.5)
plt.scatter(ori_reg, n_samples, label="out_reg", s=10, color="red", alpha=0.5)
plt.yticks((np.arange(min(ori_gt)-1, max(ori_gt)+2, 0.75)))
plt.xlabel('number of samples')
plt.ylabel('orientation in degrees')
plt.title('Orientation Prediction Model Output')
plt.legend()
plt.show()
