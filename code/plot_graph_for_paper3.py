#retieve data from firebase

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
plt.rcParams['figure.figsize'] = 8, 5


# list_task=['Task_9','Task_10','Task_11','Task_12','Task_13','Task_14','Task_15']#,'Task_13','Task_14','Task_15']
# list_task=['Task_1','Task_2','Task_3','Task_4','Task_5','Task_6','Task_7','Task_8','Task_9','Task_10','Task_11','Task_12','Task_13','Task_14','Task_15']
list_task=['Task_9']
# list_model=[3,4,5]

list_model=[2]
list_hour=[x for x in range(0,24)]

# list_method=['mean_std_005','mean_std_01', 'mean_std_015', 'mean_std_020', 'mean_std_025', 'QR_005', 'QR_01', 'QR_015', 'QR_020', 'QR_025']



# for task in list_task:

task='Task_9'

actual_price = firebase.get('/GEFcom2014_spike/{}'.format(task), 'actual_price')
plt.figure(1)

fig, ax = subplots()
ax.plot(list(np.arange(0, 24, 1)), actual_price, '-k*', alpha=0.8, label='actual_price')

method='QR_005'

list_LB = firebase.get('/GEFcom2014/{}/results/model-2/{}'.format(task, method),
                      'lower_bound')
list_UB = firebase.get('/GEFcom2014/{}/results/model-2/{}'.format(task, method),
                       'upper_bound')

# ax.plot(list(np.arange(0, 24, 1)), list_UB, '-r*', alpha=0.8, label='upper{}'.format(method))
# ax.plot(list(np.arange(0, 24, 1)), list_LB, '-r*', alpha=0.8, label='lower{}'.format(method))
ax.fill_between(list(np.arange(0, 24, 1)), list_UB, list_LB, facecolor='slategrey', alpha=0.2, label='non-price spike {}'.format(method))


method='QR_005'
list_LB = firebase.get('/GEFcom2014_spike/{}/results/model-2/{}'.format(task, method),
                       'lower_bound')
list_UB = firebase.get('/GEFcom2014_spike/{}/results/model-2/{}'.format(task, method),
                       'upper_bound')
# ax.plot(list(np.arange(0, 24, 1)), list_UB, '-g*', alpha=0.8,)
# ax.plot(list(np.arange(0, 24, 1)), list_LB, '-g*', alpha=0.8,)
ax.fill_between(list(np.arange(0, 24, 1)), list_UB, list_LB, color='grey', alpha=0.2, hatch='//', label='price spike {} '.format(method))

ax.legend(loc='best')
ax.set_xlabel('Hours')
ax.set_ylabel('$/MWhr')
plt.title('{} Interval prediction using {}'.format(task,method))
# plt.savefig('{}-compare_between_non-spike_and_spike'.format(task))
plt.show()


list_task=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
task_name=['Task_1','Task_2','Task_3','Task_4','Task_5','Task_6','Task_7','Task_8','Task_9','Task_10','Task_11','Task_12','Task_13','Task_14','Task_15']

method='QR_005'
plt.figure(2)

for num_task in list_task[3:]:
    actual_price = firebase.get('/GEFcom2014_spike/{}'.format(task_name[num_task]), 'actual_price')

    plt.subplot(4, 3, num_task - 2)
    plt.plot(list(np.arange(0, 24, 1)), actual_price, label="real_price")

    list_LB = firebase.get('/GEFcom2014_spike/{}/results/model-2/{}'.format(task_name[num_task], method),
                           'lower_bound')
    list_UB = firebase.get('/GEFcom2014_spike/{}/results/model-2/{}'.format(task_name[num_task], method),
                           'upper_bound')
    plt.fill_between(list(np.arange(0, 24, 1)), list_UB, list_LB, color='grey', alpha=0.2, hatch='//',
                    label='price spike {} '.format(method))

    plt.title('h{}'.format(num_task + 1))
    plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.35)
plt.legend(loc='best')

plt.savefig('All_task_with_spike_price_{}'.format(method))

# plt.show()
