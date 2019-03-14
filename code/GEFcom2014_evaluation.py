#retieve PICP, NMWP, CWC, pinball loss




import pandas as pd
import numpy as np
import neurolab as nl
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show

from firebase import firebase
firebase = firebase.FirebaseApplication('https://thesis-10ad5.firebaseio.com', None)


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"]=10.0
plt.rcParams['figure.figsize'] = 5, 5


# list_task=['Task_9','Task_10','Task_11','Task_12','Task_13','Task_14','Task_15']#,'Task_13','Task_14','Task_15']
list_task=['Task_1','Task_2','Task_3','Task_4','Task_5','Task_6','Task_7','Task_8','Task_9','Task_10','Task_11','Task_12','Task_13','Task_14','Task_15']
# list_task=['Task_2','Task_5']
# list_model=[3,4,5]

list_model=[2,3,4,5]

# list_method=['mean_std_005','mean_std_01','QR_005','QR_01']
# list_method=['mean_std_005', 'mean_std_01', 'mean_std_015', 'mean_std_020', 'mean_std_025', 'QR_005', 'QR_01', 'QR_015', 'QR_020', 'QR_025']
list_method=['mean_std_005','mean_std_01', 'mean_std_015', 'mean_std_020', 'mean_std_025', 'QR_005', 'QR_01', 'QR_015', 'QR_020', 'QR_025']
# list_method=['mean_std_005', 'mean_std_01','mean_std_015','mean_std_020', 'mean_std_025']

# list_method=['QR_025']

df_PICP=pd.DataFrame()
df_MPIW=pd.DataFrame()
df_NPIW=pd.DataFrame()
df_CWC=pd.DataFrame()
df_avg_pinball=pd.DataFrame()

model=2

spike_price=1
for method in list_method:
    print(method)
    list_PICP, list_MPIW, list_NPIW, list_CWC, list_avg_pinball = [], [], [], [], []
    for task in list_task:
        if spike_price==1:
            result=firebase.get('/GEFcom2014_spike/{}/results/model-{}'.format(task,model), '{}'.format(method))
        else:
            result=firebase.get('/GEFcom2014/{}/results/model-{}'.format(task,model), '{}'.format(method))
        list_PICP.append(result['PICP'])
        list_MPIW.append(result['MPIW'])
        list_NPIW.append(result['NPIW'])
        list_CWC.append(result['CWC'])
        list_avg_pinball.append(result['avg_pinball'])

    df_PICP['{}'.format(method)] = list_PICP
    df_MPIW['{}'.format(method)] = list_MPIW
    df_NPIW['{}'.format(method)] = list_NPIW
    df_CWC['{}'.format(method)] = list_CWC
    df_avg_pinball['{}'.format(method)] = list_avg_pinball


print('average CWC for mean varience method is ')
print(df_CWC.iloc[:,0:5].mean())

print('average CWC for Quantile regression method is ')
print(df_CWC.iloc[:,5:].mean())

df_avg_pinball['avg']=df_avg_pinball.mean(axis=1)
df_avg_pinball['avg_MV']=df_avg_pinball.iloc[:,0:5].mean(axis=1)
df_avg_pinball['avg_QR']=df_avg_pinball.iloc[:,5:].mean(axis=1)

print('average CWC for each method')
print(df_CWC.mean())

print(df_NPIW.mean())
print(df_PICP)
