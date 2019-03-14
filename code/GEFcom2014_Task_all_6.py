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
# list_method=['mean_std_005','mean_std_01', 'mean_std_015', 'mean_std_020', 'mean_std_025', 'QR_005', 'QR_01', 'QR_015', 'QR_020', 'QR_025']
# list_method=['mean_std_005', 'mean_std_01','mean_std_015','mean_std_020', 'mean_std_025']

list_method=['QR_025']

for num_task in list(np.arange(0,len(list_task),1)):
    actual_price = firebase.get('/GEFcom2014_spike/{}'.format(list_task[num_task]), 'actual_price')
    # for model in list_model:
    model=2
    for method in list_method:
        result=firebase.get('/GEFcom2014_spike/{}/results/model-{}'.format(list_task[num_task],model), '{}'.format(method))
        # print(result['PICP'])
        print(method)
        print(result['lower_bound'][13])
        print(result['upper_bound'][13])
        fig, ax = subplots()
        ax.plot(list(np.arange(0, 24, 1)), result['upper_bound'], '-r*', alpha=0.8, label='upper')
        ax.plot(list(np.arange(0, 24, 1)), actual_price, '-b*', alpha=0.8, label='actual_price')
        ax.plot(list(np.arange(0, 24, 1)), result['lower_bound'], '-r*', alpha=0.8, label='lower')
        ax.fill_between(list(np.arange(0, 24, 1)), result['upper_bound'], result['lower_bound'], facecolor='magenta', alpha=0.05)
        ax.legend(loc='best')
        ax.set_xlabel('Hours')
        ax.set_ylabel('$/MWhr')
        plt.title('{} Interval prediction Model-{} using {}'.format(list_task[num_task],model,method))
        plt.show()
        print('finish')
    print('finish')
print('finist')