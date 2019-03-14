
import pandas as pd
import numpy as np
import neurolab as nl
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show

import math

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"]=10.0
plt.rcParams['figure.figsize'] = 5, 5


from firebase import firebase
firebase = firebase.FirebaseApplication('https://thesis-10ad5.firebaseio.com', None)



list_task=['Task_1','Task_2','Task_3','Task_4','Task_5','Task_6','Task_7','Task_8','Task_9','Task_10','Task_11','Task_12','Task_13','Task_14','Task_15']
list_task_file=['Task1_P','Task2_P','Task3_P','Task4_P','Task5_P','Task6_P','Task7_P','Task8_P','Task9_P','Task10_P','Task11_P','Task12_P','Task13_P','Task14_P','Task15_P']
list_task_folder=['Task 1','Task 2','Task 3','Task 4','Task 5','Task 6','Task 7','Task 8','Task 9','Task 10','Task 11','Task 12','Task 13','Task 14','Task 15']
# list_method=['mean_std_005','mean_std_01','QR_005','QR_01','QR_025']

list_method=['mean_std_005']

for num_task in list(np.arange(0,len(list_task),1)):
    task = list_task[num_task]
    task_file = list_task_file[num_task]
    task_folder = list_task_folder[num_task]
    print(task)
    for model in list(np.arange(2,6,1)):
        print(model)
        for method in list_method:
            print(method)
            lower_bound = firebase.get('/GEFcom2014_spike/{}/results/model-{}/{}'.format(list_task[num_task], model,method), 'lower_bound')
            upper_bound = firebase.get('/GEFcom2014_spike/{}/results/model-{}/{}'.format(list_task[num_task], model,method), 'upper_bound')
            training_time = firebase.get('/GEFcom2014_spike/{}/results/model-{}/{}'.format(list_task[num_task], model,method), 'training_time')

            firebase.put('GEFcom2014_spike/{}/results/model-{}/mean_std_025'.format(task, model,method), 'lower_bound', lower_bound)
            firebase.put('GEFcom2014_spike/{}/results/model-{}/mean_std_025'.format(task, model,method), 'upper_bound', upper_bound)
            firebase.put('GEFcom2014_spike/{}/results/model-{}/mean_std_025'.format(task, model,method), 'training_time', training_time)

print('finish')