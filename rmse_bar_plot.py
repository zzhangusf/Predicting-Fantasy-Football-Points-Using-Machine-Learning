"""
This program reads two csv files, rmse.csv and FantasyData_rmse.csv, and
construct bar plots to compare prediciton RMSE and FantasyData RMSE.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Read prediction RMSEs and FantasayData's RMSEs
Find min RMSE for each position
"""

path = "data/"
pred = pd.read_csv(path + "rmse.csv")
fantasy = pd.read_csv(path + "FantasyData_rmse.csv")

print ("The best estimator for each position with the lowest RMSE is: \n")

RMSE = []
pred_rmse = pred.ix[:, 1:]
for column in pred_rmse:
    min_rmse = min(pred_rmse[column])
    min_rmse_index = pred_rmse[column].argmin()
    min_rmse_algo = pred.ix[min_rmse_index, 0]
    print "{}:   {}".format(min_rmse_algo, min_rmse)
    RMSE.append(min_rmse)

Fantasy_RMSE = []
for position in pred.columns.values[1: ]:
    Fantasy_RMSE.append(float(fantasy.ix[fantasy['Pos'] == position, 1]))

"""
Bar plot to compare prediction RMSE to Fantasydata.com
"""

def autolabel(rects):
    # attach value labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, 1.05 * height,
                '{0:.3f}'.format(height),
                ha='center', va='bottom')

ind = np.arange(5)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(1)
rects1 = ax.bar(ind, RMSE, width, color='r')
rects2 = ax.bar(ind + width, Fantasy_RMSE, width, color='y')
ax.set_ylabel('RMSE')
ax.set_ylim([0, 13])
ax.set_title('RMSE Comparision by Position')
ax.set_xticks(ind + width)
ax.set_xticklabels(pred.columns.values[1: ])
ax.legend((rects1[0], rects2[0]), ('Prediction', 'Fantasydata.com'))
autolabel(rects1)
autolabel(rects2)
plt.show()
