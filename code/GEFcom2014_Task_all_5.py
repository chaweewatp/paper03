#integrate high price, extreamly high price

import pandas as pd
import numpy as np
import neurolab as nl
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show
import matplotlib.gridspec as gridspec
import math
from firebase import firebase

firebase = firebase.FirebaseApplication('https://thesis-10ad5.firebaseio.com', None)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"]=10.0
plt.rcParams['figure.figsize'] = 5, 5



def get_train_spike_data(inp_df, max_total_load, max_zonal_load):
    input = np.asarray([[inp_df.iloc[ind,0]/max_total_load, inp_df.iloc[ind,1]/max_zonal_load ] for ind in inp_df.index])
    target_1 = np.asarray([[1, 0, 0] if price == 0 else [0, 1, 0] if price ==1 else [0, 0, 1] for price in inp_df['price type']])
    return input, target_1

def get_test_spike_data(inp_df, max_total_load, max_zonal_load):
    output = np.asarray([[inp_df.iloc[ind,0]/max_total_load, inp_df.iloc[ind,1]/max_zonal_load ] for ind in inp_df.index])
    return output

def get_sim_spike_data(net, inp):
    temp_output = net.sim(inp)
    output=[np.argmax(item) for item in list(temp_output)]
    return output

def get_train_data(inp_df, max_total_load, max_zonal_load, max_zonal_price, q1, q2):
    input = np.asarray([[inp_df.iloc[ind,0]/max_total_load, inp_df.iloc[ind,1]/max_zonal_load ] for ind in inp_df.index])
    target_1 = np.asarray([[inp_df['Zonal Price'].mean()/max_zonal_price, inp_df['Zonal Price'].std()/max_zonal_price] for ind in inp_df.index])
    target_2 = np.asarray([[inp_df['Zonal Price'].quantile(q1)/max_zonal_price, inp_df['Zonal Price'].quantile(q2)/max_zonal_price] for ind in inp_df.index])
    return input, target_1, target_2

def get_test_data(array1, max_total_load, max_zonal_load):
    output = np.asarray([[array1[0]/max_total_load, array1[1]/max_zonal_load]])
    return output
def get_sim_data(net, inp, max_zonal_price):
    temp_output = net.sim(inp)
    output1=[item[0]*max_zonal_price for item in temp_output]
    output2=[item[1]*max_zonal_price for item in temp_output]
    return output1, output2

def get_range_target_value(temp_df):
    r = temp_df['Zonal Price'].max() - temp_df['Zonal Price'].min()
    return r

def PI_consutruct_mean_var(var_mean, var_std, a):
    z={0.005: 2.576, 0.010:2.326, 0.025:1.960, 0.05:1.645, 0.1:1.282, 0.15:1.44, 0.2:1.282,0.25:1.142}
    upper=[var_mean[ind]+z[a]*var_std[ind] for ind in list(np.arange(0,len(var_mean),1))]
    lower=[var_mean[ind]-z[a]*var_std[ind] for ind in list(np.arange(0,len(var_mean),1))]
    return upper, lower


def PI_construct_quantile(lower_bound, upper_bound):
    upper = upper_bound
    lower = lower_bound
    return lower, upper

def cal_PI(upper, lower, actual_price,r, mu):
    number_of_price=len(actual_price)
    temp_df=pd.DataFrame()
    temp_df['upper']=upper
    temp_df['lower']=lower
    temp_df['actual price']=actual_price
    temp_df['actual price < upper']=[1 if temp_df.iloc[ind, 0] >= temp_df.iloc[ind, 2] else 0 for ind in list(temp_df.index)]
    temp_df['actual price > lower']=[1 if temp_df.iloc[ind, 1] <= temp_df.iloc[ind, 2] else 0 for ind in list(temp_df.index)]
    temp_df['within boundary']=[1 if temp_df.iloc[ind,3] == temp_df.iloc[ind,4] else 0 for ind in list(temp_df.index)]
    # print(temp_df)
    PICP=temp_df['within boundary'].sum()/number_of_price
    temp_df['upper - lower']=temp_df['upper']-temp_df['lower']
    MPIW = temp_df['upper - lower'].sum()/number_of_price
    NPIW = MPIW/r
    # print(temp_df['actual price'].max()-temp_df['actual price'].min())
    if PICP < mu:
        g = 1
    else:
        g = 0
    n=1
    CWC = NPIW*(1 + g * ((math.e)**(-n*(PICP-mu))))
    return temp_df, PICP, MPIW, NPIW, CWC


list_task=['Task_1','Task_2','Task_3','Task_4','Task_5','Task_6','Task_7','Task_8','Task_9','Task_10','Task_11','Task_12','Task_13','Task_14','Task_15']
list_task_file=['Task1_P','Task2_P','Task3_P','Task4_P','Task5_P','Task6_P','Task7_P','Task8_P','Task9_P','Task10_P','Task11_P','Task12_P','Task13_P','Task14_P','Task15_P']
list_task_folder=['Task 1','Task 2','Task 3','Task 4','Task 5','Task 6','Task 7','Task 8','Task 9','Task 10','Task 11','Task 12','Task 13','Task 14','Task 15']
# list_task=['Task_4','Task_2','Task_3','Task_4','Task_5','Task_6','Task_7','Task_8','Task_9','Task_10','Task_11','Task_12','Task_13','Task_14','Task_15']
# list_task_file=['Task4_P','Task2_P','Task3_P','Task4_P','Task5_P','Task6_P','Task7_P','Task8_P','Task9_P','Task10_P','Task11_P','Task12_P','Task13_P','Task14_P','Task15_P']
# list_task_folder=['Task 4','Task 2','Task 3','Task 4','Task 5','Task 6','Task 7','Task 8','Task 9','Task 10','Task 11','Task 12','Task 13','Task 14','Task 15']

list_model=[2]
# list_method=['mean_std_005', 'mean_std_01', 'QR_005', 'QR_01']
# list_method=['QR_01']
list_method=['mean_std_025']

for num_task in list(np.arange(0,len(list_task),1)):
    print('--------------{}-----------'.format(num_task))
    task=list_task[num_task]
    task_file=list_task_file[num_task]
    task_folder=list_task_folder[num_task]

    df1=pd.read_csv('GEFCom2014_Data/Price/{}/{}.csv'.format(task_folder, task_file))

    #separate timestamp column into two date and time column
    df1['Date'], df1['Time'] = df1['timestamp'].str.split(' ', 1).str
    df1['Hour'] = np.asarray([int(str.replace(':00','')) for str in list(df1['Time'])])
    df1=df1.drop(columns=['timestamp','Time','ZONEID'])

    #add column day of week {'Saturday':0, 'Sunday':1, ..., 'Friday':6}  when 01012011 is Saturday
    df1['day_type']=np.asarray([item//24%7 for item in list(np.arange(0,len(df1.index),1))])

    #add column holiday (0 is non-holiday, 1 is holiday)
    set_holiday=set(['01012011', '16012011', '20022011', '29052011', '04072011', '04092011', '09102011', '11112011', '25112011',
                 '25122011', '31122011', '01012012', '15012012', '19022012', '27052012', '04072012', '02092012', '14102012',
                 '11112012', '22112012', '25122012', '31122012', '01012013', '20012013', '17022013', '26052013', '04072013',
                 '01092013', '13102013', '11112013', '28112013', '25122013', '31122013'])
    #set weekend (day_type: Sat and Sun)
    # Saturday ==0
    # Sunday ==1
    set_weekend = set(df1.iloc[list(set(np.where(df1['day_type']==0)[0]).union(set(np.where(df1['day_type']==1)[0]))),3])
    df1['holiday']=[1 if date in list(set_holiday.union(set_weekend)) else 0 for date in df1['Date']]
    dict_season={'01':'Winter', '02':'Winter', '03':'Spring', '04':'Spring', '05':'Spring', '06':'Summer', '07':'Summer', '08':'Summer', '09':'Summer', '10':'Autumn','11':'Autumn','12':'Autumn'}
    df1['season'] = np.asarray([dict_season[date[0:2]] for date in list(df1['Date'])])
    forecast_df = df1[pd.isnull(df1).any(axis=1)].copy()
    forecast_df= forecast_df.reset_index(drop=True)
    df1=df1.dropna()
    r = get_range_target_value(df1)


    day_type=list(set(forecast_df['day_type']))[0]      #0-Sat, 1-Sun,,,    #model-III
    is_holiday=list(set(forecast_df['holiday']))[0]                         #model-IV
    season = list(set(forecast_df['season']))[0]                            #model-V

    price={'Task_1':[34.02, 28.26,24.82,22.9,21.59,21.33,21.55,22.49,26.25,28.95,32.17,32.44,34.25,35.97,38.05,37.77,40.87,37.51,35.44,35.28,37.55,38.41,35,32.31],
           'Task_2':[33.8,34.24,30.17,27.4,27.22,29.29,33.37,35.48,42.09,45.25,48.05,49.32,50.51,51.7,54.01,58.36,58.97,53.08,49.54,47.87,47.1,45.27,45.07,40],
           'Task_3':[38.21,33.66,30.81,29.29,30.2,32.49,37.17,48.54,52.05,52.99,56.7,61,70.58,76.75,94.74,100.77,113.37,96.91,74.8,65.9,63.91,62.8,54,48.65],
           'Task_4':[39.29,31.85,26.91,24.9,24.8,24.63,27.94,30.35,32.97,38.86,45.96,50.29,52,59.61,63.19,73.77,77.67,70.25,57.26,52.86,51.65,50.73,45.55,40.11],
           'Task_5':[49.72,42.62,39.32,38.53,39.02,38.54,39.11,47.08,50.81,56.7,61.64,67.63,74.11,80.36,85.93,108.56,109,81.4,72.1,62.14,60.91,59.4,55.99,52.9],
           'Task_6':[38.94,32.69,28.62,29,22.6,22.14,22.9,32.11,34.79,36.14,42.3,49.17,50.06,51.08,51.19,52.01,53.43,52.96,50.13,47.86,47.73,47.19,43.87,42.96],
           'Task_7':[52.6,55.79,42.94,41.96,40.74,41.54,43.1,50.63,58.59,65.72,77.44,93.76,109.34,125.56,138,164.02,175.02,155,130.17,100.53,86.06,81.91,67.74,58.61],
           'Task_8':[63.45,59.55,57.26,50.85,48.73,51.53,54.57,68.72,76.62,92.69,135.3,155.99,172.77,226.22,262.19,292.06,293.49,264.13,202.6,154.59,132.14,123.12,88.12,71.33],
           'Task_9':[64.38, 63.02, 61.28, 55.38, 50.31,50.09,57.11,62.43,73.29,84.72,130,159.77,192.58,265,291.01,313.7,318.47,288.37,178.28,148.64,107.45,90.65,83.81,72.81],
           'Task_10':[61.49,58.79,56.62,53.76,47.6,43.56,44.1,48.14,60.55,66.24,77.81,84.9,103.51,110.2,128.12,128.33,119.25,107.92,85.53,78.76,76.8,77.11,74.38,63.15],
           'Task_11':[46.4,39.41,34.94,30.69,30.62,32.81,36.04,41.56,50.23,52.74,57.4,57.94,59.31,60.91,62.42,63.01,62.98,62.62,58.58,56.62,54.34,50.93,49.42,45.28],
           'Task_12':[40.3,37.93,31.1,30.86,25.87,28.63,35.45,38.96,42.86,47.69,54.39,54.14,57.55,59.24,61.06,71.06,65.69,59.51,54.17,51.18,50.49,49.01,46.08,43.22],
           'Task_13': [42.33,38.3,38.45,36.62,35.5,38.15,40,41.55,47.64,49.67,50.55,52.7,50.3,48.39,45.85,42.79,55.09,77.82,70.24,61.78,57.26,52.29,50.48,45.6],
           'Task_14': [46.5,42.59,41.21,40.48,40.72,40.7,40.29,39.79,43.32,46.34,46.78,46.87,44.69,44.33,44.64,43.94,55.06,80.55,72.64,61.48,55.15,51.07,44.3,40.93],
           'Task_15': [66.15,71.85,69.78,69.68,64.56,68.87,96.34,113.22,108.14,97.22,102,112.69,90.41,88.66,85.22,100,124.37,161.95,126.25,113.92,107.26,89.02,85.4,86.13]}

    #create column type_price {0:'normal price', 1:'high price', 2:'spike price'}
    mu=df1['Zonal Price'].mean()
    sd=df1['Zonal Price'].std()

    df1['price type'] = [0 if price < mu + 2 * sd else 1 if price < mu + 8 * sd else 2 for price in
                         list(df1['Zonal Price'])]

    input1, target1 = get_train_spike_data(df1, df1['Forecasted Total Load'].max(),
                                              df1['Forecasted Zonal Load'].max())
    net_spike = nl.net.newff([[0, 1], [0, 1]], [5, 3])
    err = net_spike.train(input1, target1, epochs=5, show=10, goal=0.01)

    actual_price = price[task]
    firebase.put('GEFcom2014_spike/{}/'.format(task), 'actual_price', actual_price)
    firebase.put('GEFcom2014_spike/{}/'.format(task), 'day_type',
                 {0: 'Saturday', 1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday'}[
                     day_type])
    firebase.put('GEFcom2014_spike/{}/'.format(task), 'is_holiday', {1: 'holiday', 0: 'working day'}[is_holiday])
    firebase.put('GEFcom2014_spike/{}/'.format(task), 'season', season)
    print('start')

    # input2 = get_test_spike_data(forecast_df, df1['Forecasted Total Load'].max(),
    #                        df1['Forecasted Zonal Load'].max())
    # forecast_df['price type']=get_sim_spike_data(net_spike, input2)


    forecast_df['price type'] = [0 if temp_price < mu + 2 * sd else 1 if temp_price < mu + 8 * sd else 2 for temp_price
                                 in
                                 price[task]]
    actual_price_type = list(forecast_df['price type'])


    for model in list_model:
        if model ==2:
            for method in list_method:
                if method == 'mean_std_005':
                    a = 0.05
                    q1 = 0.05
                    q2 = 0.95
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        #check forecasted input is normal price
                        if forecast_df.iloc[hour,8]==0: #'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type']!=0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour,8]==1: #'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour,8]==2: #'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target1, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        var_mean, var_std = get_sim_data(net, input2, df3['Zonal Price'].max())
                        upper, lower = PI_consutruct_mean_var(var_mean, var_std, a)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'mean_std_005',
                                     {'lower_bound': list_lower_01, 'upper_bound': list_upper_01, 'training_time': elapsed_time})
                elif method == 'mean_std_01':
                    a = 0.1
                    q1 = 0.05
                    q2 = 0.95
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target1, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        var_mean, var_std = get_sim_data(net, input2, df3['Zonal Price'].max())
                        upper, lower = PI_consutruct_mean_var(var_mean, var_std, a)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'mean_std_01',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})
                elif method == 'mean_std_025':
                    a = 0.25
                    q1 = 0.25
                    q2 = 0.75
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target1, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        var_mean, var_std = get_sim_data(net, input2, df3['Zonal Price'].max())
                        upper, lower = PI_consutruct_mean_var(var_mean, var_std, a)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'mean_std_025',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})

                elif method == 'QR_005':
                    a = 0.1
                    q1 = 0.05
                    q2 = 0.95
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target2, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        lower_bound, upper_bound = get_sim_data(net, input2, df3['Zonal Price'].max())
                        lower, upper= PI_construct_quantile(lower_bound, upper_bound)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'QR_005',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})
                elif method == 'QR_01':
                    a = 0.1
                    q1 = 0.1
                    q2 = 0.9
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target2, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        lower_bound, upper_bound = get_sim_data(net, input2, df3['Zonal Price'].max())
                        lower, upper = PI_construct_quantile(lower_bound, upper_bound)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'QR_01',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})
                elif method == 'QR_025':
                    print('QR_025')
                    a = 0.1
                    q1 = 0.25
                    q2 = 0.75
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target2, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        lower_bound, upper_bound = get_sim_data(net, input2, df3['Zonal Price'].max())
                        lower, upper = PI_construct_quantile(lower_bound, upper_bound)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'QR_025',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})
        elif model == 3:
            for method in list_method:
                if method == 'mean_std_005':
                    a = 0.05
                    q1 = 0.05
                    q2 = 0.95
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['day_type'] != day_type)[0])))
                        df2 = df2.reset_index(drop=True)
                        #check forecasted input is normal price
                        if forecast_df.iloc[hour,8]==0: #'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type']!=0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour,8]==1: #'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour,8]==2: #'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target1, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        var_mean, var_std = get_sim_data(net, input2, df3['Zonal Price'].max())
                        upper, lower = PI_consutruct_mean_var(var_mean, var_std, a)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'mean_std_005',
                                     {'lower_bound': list_lower_01, 'upper_bound': list_upper_01, 'training_time': elapsed_time})
                elif method == 'mean_std_01':
                    a = 0.1
                    q1 = 0.05
                    q2 = 0.95
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['day_type'] != day_type)[0])))
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target1, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        var_mean, var_std = get_sim_data(net, input2, df3['Zonal Price'].max())
                        upper, lower = PI_consutruct_mean_var(var_mean, var_std, a)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'mean_std_01',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})
                elif method == 'QR_005':
                    a = 0.1
                    q1 = 0.05
                    q2 = 0.95
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['day_type'] != day_type)[0])))
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target2, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        lower_bound, upper_bound = get_sim_data(net, input2, df3['Zonal Price'].max())
                        lower, upper= PI_construct_quantile(lower_bound, upper_bound)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                        print('finish hour')
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'QR_005',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})
                elif method == 'QR_01':
                    a = 0.1
                    q1 = 0.1
                    q2 = 0.9
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['day_type'] != day_type)[0])))
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target2, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        lower_bound, upper_bound = get_sim_data(net, input2, df3['Zonal Price'].max())
                        lower, upper= PI_construct_quantile(lower_bound, upper_bound)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'QR_01',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})
        elif model == 4:
            for method in list_method:
                if method == 'mean_std_005':
                    a = 0.05
                    q1 = 0.05
                    q2 = 0.95
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['day_type'] != day_type)[0])))
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['holiday'] != is_holiday)[0])))
                        df2 = df2.reset_index(drop=True)
                        #check forecasted input is normal price
                        if forecast_df.iloc[hour,8]==0: #'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type']!=0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour,8]==1: #'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour,8]==2: #'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target1, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        var_mean, var_std = get_sim_data(net, input2, df3['Zonal Price'].max())
                        upper, lower = PI_consutruct_mean_var(var_mean, var_std, a)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'mean_std_005',
                                     {'lower_bound': list_lower_01, 'upper_bound': list_upper_01, 'training_time': elapsed_time})
                elif method == 'mean_std_01':
                    a = 0.1
                    q1 = 0.05
                    q2 = 0.95
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['day_type'] != day_type)[0])))
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['holiday'] != is_holiday)[0])))
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target1, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        var_mean, var_std = get_sim_data(net, input2, df3['Zonal Price'].max())
                        upper, lower = PI_consutruct_mean_var(var_mean, var_std, a)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'mean_std_01',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})
                elif method == 'QR_005':
                    a = 0.1
                    q1 = 0.05
                    q2 = 0.95
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['day_type'] != day_type)[0])))
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['holiday'] != is_holiday)[0])))
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target2, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        lower_bound, upper_bound = get_sim_data(net, input2, df3['Zonal Price'].max())
                        lower, upper = PI_construct_quantile(lower_bound, upper_bound)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'QR_005',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})
                elif method == 'QR_01':
                    a = 0.1
                    q1 = 0.1
                    q2 = 0.9
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['day_type'] != day_type)[0])))
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['holiday'] != is_holiday)[0])))
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target2, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        lower_bound, upper_bound = get_sim_data(net, input2, df3['Zonal Price'].max())
                        lower, upper = PI_construct_quantile(lower_bound, upper_bound)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                        print('finish hour')
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'QR_01',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})
        elif model == 5:
            for method in list_method:
                if method == 'mean_std_005':
                    a = 0.05
                    q1 = 0.05
                    q2 = 0.95
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['day_type'] != day_type)[0])))
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['holiday'] != is_holiday)[0])))
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['season'] != season)[0])))
                        df2 = df2.reset_index(drop=True)
                        #check forecasted input is normal price
                        if forecast_df.iloc[hour,8]==0: #'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type']!=0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour,8]==1: #'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour,8]==2: #'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target1, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        var_mean, var_std = get_sim_data(net, input2, df3['Zonal Price'].max())
                        upper, lower = PI_consutruct_mean_var(var_mean, var_std, a)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'mean_std_005',
                                     {'lower_bound': list_lower_01, 'upper_bound': list_upper_01, 'training_time': elapsed_time})
                elif method == 'mean_std_01':
                    a = 0.1
                    q1 = 0.05
                    q2 = 0.95
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['day_type'] != day_type)[0])))
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['holiday'] != is_holiday)[0])))
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['season'] != season)[0])))
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target1, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        var_mean, var_std = get_sim_data(net, input2, df3['Zonal Price'].max())
                        upper, lower = PI_consutruct_mean_var(var_mean, var_std, a)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'mean_std_01',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})
                elif method == 'QR_005':
                    a = 0.1
                    q1 = 0.05
                    q2 = 0.95
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['day_type'] != day_type)[0])))
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['holiday'] != is_holiday)[0])))
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['season'] != season)[0])))
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target2, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        lower_bound, upper_bound = get_sim_data(net, input2, df3['Zonal Price'].max())
                        lower, upper = PI_construct_quantile(lower_bound, upper_bound)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'QR_005',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})
                elif method == 'QR_01':
                    a = 0.1
                    q1 = 0.1
                    q2 = 0.9
                    list_upper_01 = []
                    list_lower_01 = []
                    for hour in list(np.arange(0, 24, 1)):
                        print(hour)
                        df2 = df1.drop(list(set(np.where(df1['Hour'] != hour)[
                                                    0]))).copy()  # drop column which day_type not equal to 0 or select only day_type =0
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['day_type'] != day_type)[0])))
                        df2 = df2.reset_index(drop=True)
                        df2 = df2.drop(list(set(np.where(df2['holiday'] != is_holiday)[0])))
                        df2 = df2.reset_index(drop=True)
                        df2=df2.drop(list(set(np.where(df2['season']!= season)[0])))
                        df2 = df2.reset_index(drop=True)
                        # check forecasted input is normal price
                        if forecast_df.iloc[hour, 8] == 0:  # 'normal price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 0)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 1:  # 'high price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 1)[0]))).copy()
                            df3 = df3.reset_index(drop=True)
                        elif forecast_df.iloc[hour, 8] == 2:  # 'spike price'
                            df3 = df2.drop(list(set(np.where(df2['price type'] != 2)[0]))).copy()
                            df3 = df3.reset_index(drop=True)

                        input1, target1, target2 = get_train_data(df3, df3['Forecasted Total Load'].max(),
                                                                  df3['Forecasted Zonal Load'].max(),
                                                                  df3['Zonal Price'].max(), q1, q2)
                        net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
                        start_time = time.time()
                        err = net.train(input1, target2, epochs=500, show=10, goal=0.01)
                        elapsed_time = time.time() - start_time  # again taking current time - starting time

                        input2 = get_test_data(np.asarray(forecast_df.loc[hour]), df3['Forecasted Total Load'].max(),
                                               df3['Forecasted Zonal Load'].max())

                        lower_bound, upper_bound = get_sim_data(net, input2, df3['Zonal Price'].max())
                        lower, upper = PI_construct_quantile(lower_bound, upper_bound)
                        list_upper_01.append(upper[0])
                        list_lower_01.append(lower[0])
                    firebase.put('GEFcom2014_spike/{}/results/model-{}'.format(task, model), 'QR_01',
                                 {'lower_bound': list_lower_01, 'upper_bound': list_upper_01,
                                  'training_time': elapsed_time})

print('finish')





#f
# plotting
# x = df1['Forecasted Total Load']
# y = df1['Zonal Price']
#
# fig = plt.figure(figsize=(8, 8))
# gs = gridspec.GridSpec(3, 3)
# ax_main = plt.subplot(gs[1:3, :2])
# ax_xDist = plt.subplot(gs[0, :2], sharex=ax_main)
# ax_yDist = plt.subplot(gs[1:3, 2], sharey=ax_main)
#
# ax_main.scatter(x, y, marker='.', c=df1['Hour'], alpha=0.2)
# ax_main.set(xlabel="x data", ylabel="y data")
#
# ax_xDist.hist(x, bins=100, align='mid')
# ax_xDist.set(ylabel='count')
# ax_xCumDist = ax_xDist.twinx()
# ax_xCumDist.hist(x, bins=100, cumulative=True, histtype='step', normed=True, color='r', align='mid')
# ax_xCumDist.tick_params('y', colors='r')
# ax_xCumDist.set_ylabel('cumulative', color='r')
#
# ax_yDist.hist(y, bins=100, orientation='horizontal', align='mid')
# ax_yDist.set(xlabel='count')
# ax_yCumDist = ax_yDist.twiny()
# ax_yCumDist.hist(y, bins=100, cumulative=True, histtype='step', normed=True, color='r', align='mid',
#                  orientation='horizontal')
# ax_yCumDist.tick_params('x', colors='r')
# ax_yCumDist.set_xlabel('cumulative', color='r')
# plt.show()
#
#
#
# x = df1['Forecasted Zonal Load']
# y = df1['Zonal Price']
#
# fig = plt.figure(figsize=(8, 8))
# gs = gridspec.GridSpec(3, 3)
# ax_main = plt.subplot(gs[1:3, :2])
# ax_xDist = plt.subplot(gs[0, :2], sharex=ax_main)
# ax_yDist = plt.subplot(gs[1:3, 2], sharey=ax_main)
#
# ax_main.scatter(x, y, marker='.', c=df1['Hour'], alpha=0.2)
# ax_main.set(xlabel="x data", ylabel="y data")
#
# ax_xDist.hist(x, bins=100, align='mid')
# ax_xDist.set(ylabel='count')
# ax_xCumDist = ax_xDist.twinx()
# ax_xCumDist.hist(x, bins=100, cumulative=True, histtype='step', normed=True, color='r', align='mid')
# ax_xCumDist.tick_params('y', colors='r')
# ax_xCumDist.set_ylabel('cumulative', color='r')
#
# ax_yDist.hist(y, bins=100, orientation='horizontal', align='mid')
# ax_yDist.set(xlabel='count')
# ax_yCumDist = ax_yDist.twiny()
# ax_yCumDist.hist(y, bins=100, cumulative=True, histtype='step', normed=True, color='r', align='mid',
#                  orientation='horizontal')
# ax_yCumDist.tick_params('x', colors='r')
# ax_yCumDist.set_xlabel('cumulative', color='r')
# plt.show()
