import numpy as np
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

file_dfm = '/data2/law/tcl/result_dfm.txt'
file_recce = '/data2/law/tcl/result_recce.txt'
dfm_data = []
dfm_label = []
recce_data = []
recce_label = []
with open (file_dfm,'r') as fo:
    for line in fo:
        line=line.strip().split(' ')
        dfm_data.append(float(line[0]))
        dfm_label.append(int(line[1]))

with open (file_recce,'r') as fo:
    for line in fo:
        line=line.strip().split(' ')
        recce_data.append(float(line[0][1:-1]))
        recce_label.append(int(line[1]))

# dfm_data = np.array(dfm_data)
# recce_data = np.array(recce_data)
FPR0,TPR0,threshold0=roc_curve(dfm_label,dfm_data,pos_label=1)
FPR1,TPR1,threshold1=roc_curve(recce_label,recce_data,pos_label=1)

AUC0=auc(FPR0,TPR0)
AUC1=auc(FPR1,TPR1)
print(AUC0,'---',AUC1)

plt.figure()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.plot(FPR0,TPR0,color='g')
plt.plot(FPR1,TPR1,color='r')
plt.plot([0, 1], [0, 1], color='m', linestyle='--')
plt.show()


