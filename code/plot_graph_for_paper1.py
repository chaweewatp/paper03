#retieve data from firebase and saving to CSV file

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
# list_model=[3,4,5]

list_model=[2]
list_hour=[x for x in range(0,24)]

# # list_method=['mean_std_005','mean_std_01','QR_005','QR_01']
# # list_method=['mean_std_005']
# # list_method=['mean_std_01']
# # list_method=['QR_005']
# list_method=['QR_01']
#
#
#
#
#
# df_MV_005_UB=pd.DataFrame()
# df_MV_010_UB=pd.DataFrame()
# df_QR_005_UB=pd.DataFrame()
# df_QR_010_UB=pd.DataFrame()
# df_MV_005_UB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_MV_010_UB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_QR_005_UB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_QR_010_UB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_MV_005_LB=pd.DataFrame()
# df_MV_010_LB=pd.DataFrame()
# df_QR_005_LB=pd.DataFrame()
# df_QR_010_LB=pd.DataFrame()
# df_MV_005_LB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_MV_010_LB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_QR_005_LB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_QR_010_LB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#
#
# for hour in list_hour:
#     print(hour)
#     for method in list_method:
#         print(method)
#         list_price_LB=[]
#         list_price_UB = []
#         for num_task in list(np.arange(0,len(list_task),1)):
#             # print(num_task)
#
#
#             result = firebase.get('/GEFcom2014_spike/{}/results/model-2/{}/lower_bound'.format(list_task[num_task], method),
#                               '{}'.format(hour))
#             list_price_LB.append(result)
#             result = firebase.get(
#                 '/GEFcom2014_spike/{}/results/model-2/{}/upper_bound'.format(list_task[num_task], method),
#                 '{}'.format(hour))
#             list_price_UB.append(result)
#         # print(list_price)
#         if method =='mean_std_005':
#             df_MV_005_LB['h{}'.format(hour)]=list_price_LB
#             df_MV_005_UB['h{}'.format(hour)]=list_price_UB
#
#         elif method =='mean_std_01':
#             df_MV_010_LB['h{}'.format(hour)]=list_price_LB
#             df_MV_010_UB['h{}'.format(hour)]=list_price_UB
#         elif method == 'QR_005':
#             df_QR_005_LB['h{}'.format(hour)]=list_price_LB
#             df_QR_005_UB['h{}'.format(hour)]=list_price_UB
#
#         elif method =='QR_01':
#             df_QR_010_LB['h{}'.format(hour)]=list_price_LB
#             df_QR_010_UB['h{}'.format(hour)]=list_price_UB
#
# print('file saving')
# # df_MV_005_LB.to_csv('price_spike_model2_MV_005_LB.csv', sep=':', index=False)
# # df_MV_010_LB.to_csv('price_spike_model2_MV_010_LB.csv', sep=':', index=False)
# # df_QR_005_LB.to_csv('price_spike_model2_QR_005_LB.csv', sep=':', index=False)
# df_QR_010_LB.to_csv('price_spike_model2_QR_010_LB.csv', sep=':', index=False)
# # df_MV_005_UB.to_csv('price_spike_model2_MV_005_UB.csv', sep=':', index=False)
# # df_MV_010_UB.to_csv('price_spike_model2_MV_010_UB.csv', sep=':', index=False)
# # df_QR_005_UB.to_csv('price_spike_model2_QR_005_UB.csv', sep=':', index=False)
# df_QR_010_UB.to_csv('price_spike_model2_QR_010_UB.csv', sep=':', index=False)
#
#
#
#
# df_MV_005_UB=pd.DataFrame()
# df_MV_010_UB=pd.DataFrame()
# df_QR_005_UB=pd.DataFrame()
# df_QR_010_UB=pd.DataFrame()
# df_MV_005_UB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_MV_010_UB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_QR_005_UB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_QR_010_UB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_MV_005_LB=pd.DataFrame()
# df_MV_010_LB=pd.DataFrame()
# df_QR_005_LB=pd.DataFrame()
# df_QR_010_LB=pd.DataFrame()
# df_MV_005_LB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_MV_010_LB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_QR_005_LB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# df_QR_010_LB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#
# for hour in list_hour:
#     print(hour)
#     for method in list_method:
#         print(method)
#         list_price_LB=[]
#         list_price_UB = []
#         for num_task in list(np.arange(0,len(list_task),1)):
#             # print(num_task)
#
#             result = firebase.get('/GEFcom2014/{}/results/model-2/{}/lower_bound'.format(list_task[num_task], method),
#                               '{}'.format(hour))
#             list_price_LB.append(result)
#             result = firebase.get(
#                 '/GEFcom2014/{}/results/model-2/{}/upper_bound'.format(list_task[num_task], method),
#                 '{}'.format(hour))
#             list_price_UB.append(result)
#         # print(list_price)
#         if method =='mean_std_005':
#             df_MV_005_LB['h{}'.format(hour)]=list_price_LB
#             df_MV_005_UB['h{}'.format(hour)]=list_price_UB
#
#         elif method =='mean_std_01':
#             df_MV_010_LB['h{}'.format(hour)]=list_price_LB
#             df_MV_010_UB['h{}'.format(hour)]=list_price_UB
#         elif method == 'QR_005':
#             df_QR_005_LB['h{}'.format(hour)]=list_price_LB
#             df_QR_005_UB['h{}'.format(hour)]=list_price_UB
#
#         elif method =='QR_01':
#             df_QR_010_LB['h{}'.format(hour)]=list_price_LB
#             df_QR_010_UB['h{}'.format(hour)]=list_price_UB
#
# print('file saving')
#
# # df_MV_005_LB.to_csv('price_model2_MV_005_LB.csv', sep=':', index=False)
# # df_MV_010_LB.to_csv('price_model2_MV_010_LB.csv', sep=':', index=False)
# # df_QR_005_LB.to_csv('price_model2_QR_005_LB.csv', sep=':', index=False)
# df_QR_010_LB.to_csv('price_model2_QR_010_LB.csv', sep=':', index=False)
# # df_MV_005_UB.to_csv('price_model2_MV_005_UB.csv', sep=':', index=False)
# # df_MV_010_UB.to_csv('price_model2_MV_010_UB.csv', sep=':', index=False)
# # df_QR_005_UB.to_csv('price_model2_QR_005_UB.csv', sep=':', index=False)
# df_QR_010_UB.to_csv('price_model2_QR_010_UB.csv', sep=':', index=False)
#
#
#
# # df_actual_price=pd.DataFrame()
# # df_actual_price['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# #
# # for hour in list_hour:
# #     print(hour)
# #     list_price=[]
# #     for num_task in list(np.arange(0, len(list_task), 1)):
# #
# #         result = firebase.get('/GEFcom2014_spike/{}/actual_price'.format(list_task[num_task]),
# #                               '{}'.format(hour))
# #         list_price.append(result)
# #     df_actual_price['h{}'.format(hour)]=list_price
# # df_actual_price.to_csv('actual_price.csv', sep=':', index=False)


list_method=['QR_025']
df_QR_025_UB=pd.DataFrame()
df_QR_025_UB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
df_QR_025_LB=pd.DataFrame()
df_QR_025_LB['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

for hour in list_hour:
    print(hour)
    for method in list_method:
        print(method)
        list_price_LB=[]
        list_price_UB = []
        for num_task in list(np.arange(0,len(list_task),1)):
            # print(num_task)

            result = firebase.get('/GEFcom2014_spike/{}/results/model-2/{}/lower_bound'.format(list_task[num_task], method),
                              '{}'.format(hour))
            list_price_LB.append(result)
            print(result)
            result = firebase.get(
                '/GEFcom2014_spike/{}/results/model-2/{}/upper_bound'.format(list_task[num_task], method),
                '{}'.format(hour))
            list_price_UB.append(result)
        # print(list_price)
        if method =='mean_std_005':
            df_MV_005_LB['h{}'.format(hour)]=list_price_LB
            df_MV_005_UB['h{}'.format(hour)]=list_price_UB

        elif method =='mean_std_01':
            df_MV_010_LB['h{}'.format(hour)]=list_price_LB
            df_MV_010_UB['h{}'.format(hour)]=list_price_UB
        elif method == 'QR_005':
            df_QR_005_LB['h{}'.format(hour)]=list_price_LB
            df_QR_005_UB['h{}'.format(hour)]=list_price_UB

        elif method =='QR_01':
            df_QR_010_LB['h{}'.format(hour)]=list_price_LB
            df_QR_010_UB['h{}'.format(hour)]=list_price_UB

        elif method =='QR_025':
            df_QR_025_LB['h{}'.format(hour)]=list_price_LB
            df_QR_025_UB['h{}'.format(hour)]=list_price_UB

df_QR_025_LB.to_csv('price_spike_model2_QR_025_LB.csv', sep=':', index=False)
df_QR_025_UB.to_csv('price_spike_model2_QR_025_UB.csv', sep=':', index=False)

print('file saving')



print('end')


