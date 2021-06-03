import matplotlib.pyplot as plt
import csv
import numpy as np

x_gt = []
y_gt = []
x_reg = []
y_reg = []
with open("3Dtrain_gt_2_12.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for col in csv_reader:
        x_gt.append(col[0])
        y_gt.append(col[1])
x_gt = np.asarray(x_gt)
x_gt = x_gt.astype(float)
y_gt = np.asarray(y_gt)
y_gt = y_gt.astype(float)
# plt.scatter(x_gt, y_gt, label="ground_truth", color="blue")
with open("3Dtrain_regout_2_12.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for col in csv_reader:
        x_reg.append(col[0])
        y_reg.append(col[1])
x_reg = np.asarray(x_reg)
x_reg = x_reg.astype(float)
y_reg = np.asarray(y_reg)
y_reg = y_reg.astype(float)
plt.scatter(x_gt, y_gt, label="ground_truth", s=10, color="blue", alpha=0.5)
plt.scatter(x_reg, y_reg, label="out_reg", s=10, color="red", alpha=0.5)
plt.xticks((np.arange(min(x_gt)-1, max(x_gt)+2, 0.75)))
plt.yticks((np.arange(min(y_gt)-1, max(y_gt)+2, 0.75)))
plt.xlabel('x in metres')
plt.ylabel('y in metres')
plt.title('3D Visual Localization Model Output')
plt.legend()
plt.show()
