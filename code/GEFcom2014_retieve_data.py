#retieve data for plotting

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

list_method=['mean_std_005','mean_std_01','QR_005','QR_01']

[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
df_model2=pd.DataFrame()
df_model3=pd.DataFrame()
df_model4=pd.DataFrame()
df_model5=pd.DataFrame()

df_model2['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
df_model3['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
df_model4['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
df_model5['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]



for model in list_model:
    for method in list_method:
        list_PICP=[]
        list_CWC=[]
        list_lower_price=[]
        list_upper_price=[]
        for num_task in list(np.arange(0,len(list_task),1)):
            result = firebase.get('/GEFcom2014_spike/{}/results/model-{}'.format(list_task[num_task], model),
                                  '{}'.format(method))
            list_PICP.append(result['PICP'])
            # list_CWC.append(result['CWC'])
            list_lower_price.append(result['lower_bound'])
            list_upper_price.append(result['upper_bound'])
            # print(result['lower_bound'])
        if model == 2:
            df_model2['{}_PICP'.format(method)] = [round(x, 2) for x in list_PICP]
            # df_model2['{}_CWC'.format(method)]=[round(x*100,2) for x in list_CWC]
            df_model2['{}_lower_price'.format(method)] = [x for x in list_lower_price]
            df_model2['{}_upper_price'.format(method)] = [x for x in list_upper_price]
        elif model == 3:
            df_model3['{}_PICP'.format(method)]=[round(x,2) for x in list_PICP]
            df_model3['{}_CWC'.format(method)]=[round(x*100,2) for x in list_CWC]
        elif model == 4:
            df_model4['{}_PICP'.format(method)]=[round(x,2) for x in list_PICP]
            df_model4['{}_CWC'.format(method)]=[round(x*100,2) for x in list_CWC]
        elif model == 5:
            df_model5['{}_PICP'.format(method)]=[round(x,2) for x in list_PICP]
            df_model5['{}_CWC'.format(method)]=[round(x*100,2) for x in list_CWC]

df_model2.to_csv('price_spike_model2.csv', sep=':', index=False)
# df_model3.to_csv('price_spike_model3.csv', sep='&', index=False)
# df_model4.to_csv('price_spike_model4.csv', sep='&', index=False)
# df_model5.to_csv('price_spike_model5.csv', sep='&', index=False)



df_model2=pd.DataFrame()
df_model3=pd.DataFrame()
df_model4=pd.DataFrame()
df_model5=pd.DataFrame()

df_model2['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
df_model3['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
df_model4['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
df_model5['task']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]


for model in list_model:
    for method in list_method:
        list_PICP=[]
        list_CWC=[]
        list_lower_price=[]
        list_upper_price=[]
        for num_task in list(np.arange(0,len(list_task),1)):
            result = firebase.get('/GEFcom2014/{}/results/model-{}'.format(list_task[num_task], model),
                                  '{}'.format(method))
            list_PICP.append(result['PICP'])
            # list_CWC.append(result['CWC'])
            list_lower_price.append(result['lower_bound'])
            list_upper_price.append(result['upper_bound'])
            # print(result['lower_bound'])
        if model == 2:
            df_model2['{}_PICP'.format(method)]=[round(x,2) for x in list_PICP]
            # df_model2['{}_CWC'.format(method)]=[round(x*100,2) for x in list_CWC]
            df_model2['{}_lower_price'.format(method)]=[x for x in list_lower_price]
            df_model2['{}_upper_price'.format(method)]=[x for x in list_upper_price]

        elif model == 3:
            df_model3['{}_PICP'.format(method)]=[round(x,2) for x in list_PICP]
            df_model3['{}_CWC'.format(method)]=[round(x*100,2) for x in list_CWC]
        elif model == 4:
            df_model4['{}_PICP'.format(method)]=[round(x,2) for x in list_PICP]
            df_model4['{}_CWC'.format(method)]=[round(x*100,2) for x in list_CWC]
        elif model == 5:
            df_model5['{}_PICP'.format(method)]=[round(x,2) for x in list_PICP]
            df_model5['{}_CWC'.format(method)]=[round(x*100,2) for x in list_CWC]

df_model2.to_csv('price_model2.csv', sep=':', index=False)
# df_model3.to_csv('price_model3.csv', sep='&', index=False)
# df_model4.to_csv('price_model4.csv', sep='&', index=False)
# df_model5.to_csv('price_model5.csv', sep='&', index=False)
print('finish')
