#calculation PICP, MPIW, NPIW, CWC

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


def get_range_target_value(temp_df):
    r = temp_df['Zonal Price'].max() - temp_df['Zonal Price'].min()
    return r

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
# list_method=['mean_std_005','mean_std_01','QR_005','QR_01','QR_025']

list_method=['mean_std_005', 'mean_std_01', 'mean_std_015', 'mean_std_020', 'mean_std_025', 'QR_005', 'QR_01', 'QR_015', 'QR_020', 'QR_025']
# list_method=['mean_std_005']

for num_task in list(np.arange(0,len(list_task),1)):

    task=list_task[num_task]
    task_file=list_task_file[num_task]
    task_folder=list_task_folder[num_task]
    print(task)
    actual_price = firebase.get('/GEFcom2014/{}'.format(list_task[num_task]), 'actual_price')

    df1 = pd.read_csv('GEFCom2014_Data/Price/{}/{}.csv'.format(task_folder, task_file))

    # separate timestamp column into two date and time column
    df1['Date'], df1['Time'] = df1['timestamp'].str.split(' ', 1).str
    df1['Hour'] = np.asarray([int(str.replace(':00', '')) for str in list(df1['Time'])])
    df1 = df1.drop(columns=['timestamp', 'Time', 'ZONEID'])

    # add column day of week {'Saturday':0, 'Sunday':1, ..., 'Friday':6}  when 01012011 is Saturday
    df1['day_type'] = np.asarray([item // 24 % 7 for item in list(np.arange(0, len(df1.index), 1))])

    # add column holiday (0 is non-holiday, 1 is holiday)
    set_holiday = set(
        ['01012011', '16012011', '20022011', '29052011', '04072011', '04092011', '09102011', '11112011', '25112011',
         '25122011', '31122011', '01012012', '15012012', '19022012', '27052012', '04072012', '02092012', '14102012',
         '11112012', '22112012', '25122012', '31122012', '01012013', '20012013', '17022013', '26052013', '04072013',
         '01092013', '13102013', '11112013', '28112013', '25122013', '31122013'])
    # set weekend (day_type: Sat and Sun)
    # Saturday ==0
    # Sunday ==1
    set_weekend = set(
        df1.iloc[list(set(np.where(df1['day_type'] == 0)[0]).union(set(np.where(df1['day_type'] == 1)[0]))), 3])
    df1['holiday'] = [1 if date in list(set_holiday.union(set_weekend)) else 0 for date in df1['Date']]
    dict_season = {'01': 'Winter', '02': 'Winter', '03': 'Spring', '04': 'Spring', '05': 'Spring', '06': 'Summer',
                   '07': 'Summer', '08': 'Summer', '09': 'Summer', '10': 'Autumn', '11': 'Autumn', '12': 'Autumn'}
    df1['season'] = np.asarray([dict_season[date[0:2]] for date in list(df1['Date'])])
    # forecast_df = df1[pd.isnull(df1).any(axis=1)].copy()
    # forecast_df = forecast_df.reset_index(drop=True)
    df1 = df1.dropna()

    r = get_range_target_value(df1)
    r = 5
    firebase.put('GEFcom2014/{}/'.format(task), 'underlying_taget', r)

    # print(actual_price)
    #non_spike model
    print('start non-spike model')
    for model in list(np.arange(2,6,1)):
        # print(model)
        if model==2:
            for method in list_method:
                # print(method)
                lower_bound = firebase.get('/GEFcom2014/{}/results/model-{}/{}'.format(list_task[num_task], model,method), 'lower_bound')
                upper_bound = firebase.get('/GEFcom2014/{}/results/model-{}/{}'.format(list_task[num_task], model,method), 'upper_bound')
                # lower_bound = firebase.get(
                #     '/GEFcom2014_spike/{}/results/model-{}/{}'.format(list_task[num_task], model, method),
                #     'lower_bound')
                # upper_bound = firebase.get(
                #     '/GEFcom2014_spike/{}/results/model-{}/{}'.format(list_task[num_task], model, method),
                #     'upper_bound')

                if method == 'mean_std_005':
                    mu = 0.9
                    tou_u= 0.95
                    tou_l=1-tou_u
                elif method == 'mean_std_01':
                    mu = 0.8
                    tou_u= 0.9
                    tou_l=1-tou_u

                elif method == 'mean_std_015':
                    mu = 0.7
                    tou_u= 0.85
                    tou_l=1-tou_u

                elif method == 'mean_std_02':
                    mu = 0.6
                    tou_u= 0.80
                    tou_l=1-tou_u

                elif method == 'mean_std_025':
                    mu = 0.5
                    tou_u= 0.75
                    tou_l=1-tou_u

                elif method == 'QR_005':
                    mu = 0.9
                    tou_u= 0.95
                    tou_l=1-tou_u

                elif method == 'QR_01':
                    mu = 0.8
                    tou_u= 0.9
                    tou_l=1-tou_u

                elif method == 'QR_015':
                    mu = 0.7
                    tou_u= 0.85
                    tou_l=1-tou_u

                elif method == 'QR_020':
                    mu = 0.6
                    tou_u= 0.8
                    tou_l=1-tou_u

                elif method == 'QR_025':
                    mu=0.5
                    tou_u= 0.75
                    tou_l=1-tou_u

                PI_df_02, PICP, MPIW, NPIW, CWC= cal_PI(upper_bound, lower_bound, actual_price, r, mu)
                list_pinball = []
                for UB, LB, AC in zip(upper_bound, lower_bound, actual_price):
                    # if AC < LB:
                    #     pinball = (LB - AC) * (1 - tou)
                    # elif AC > UB:
                    #     pinball = (AC - UB) * tou
                    # else:
                    #     if AC > (UB + LB) / 2:
                    #         pinball = (AC - (UB + LB) / 2) * tou
                    #     else:
                    #         pinball = ((UB + LB) / 2 - AC) * (1 - tou)
                    #     pinball = 0

                    if AC < LB:
                        pinball_1 = (LB - AC) * (1 - tou_l)
                        pinball_2 = (UB - AC) * (1-tou_u)
                    elif AC > UB:
                        pinball_1 = (AC - LB) * tou_l
                        pinball_2 = (AC - UB) * tou_u
                    else:
                        pinball_1 = (AC - LB) * tou_l
                        pinball_2 = (UB - AC) * (1 - tou_u)
                        pinball_1 = 0
                        pinball_2 = 0


                    # print(pinball)
                    pinball=(pinball_1+pinball_2)/2
                    list_pinball.append(pinball)
                avg_pinball_val=np.array(list_pinball).mean()
                print(avg_pinball_val)

                firebase.put('GEFcom2014/{}/results/model-{}/{}'.format(task, model,method), 'PICP', PICP)
                firebase.put('GEFcom2014/{}/results/model-{}/{}'.format(task, model, method), 'MPIW', MPIW)
                firebase.put('GEFcom2014/{}/results/model-{}/{}'.format(task, model, method), 'NPIW', NPIW)
                firebase.put('GEFcom2014/{}/results/model-{}/{}'.format(task, model, method), 'CWC', CWC)
                firebase.put('GEFcom2014/{}/results/model-{}/{}'.format(task, model, method), 'avg_pinball', avg_pinball_val)
                # firebase.put('GEFcom2014_spike/{}/results/model-{}/{}'.format(task, model, method), 'PICP', PICP)
                # firebase.put('GEFcom2014_spike/{}/results/model-{}/{}'.format(task, model, method), 'MPIW', MPIW)
                # firebase.put('GEFcom2014_spike/{}/results/model-{}/{}'.format(task, model, method), 'NPIW', NPIW)
                # firebase.put('GEFcom2014_spike/{}/results/model-{}/{}'.format(task, model, method), 'CWC', CWC)
                # firebase.put('GEFcom2014_spike/{}/results/model-{}/{}'.format(task, model, method), 'avg_pinball',
                #              avg_pinball_val)

                # calculate pinball loss function

    #spike model
    print('start spike model')
    for model in list(np.arange(2,6,1)):
        # print(model)
        if model==2:
            for method in list_method:
                # print(method)

                lower_bound = firebase.get(
                    '/GEFcom2014_spike/{}/results/model-{}/{}'.format(list_task[num_task], model, method),
                    'lower_bound')
                upper_bound = firebase.get(
                    '/GEFcom2014_spike/{}/results/model-{}/{}'.format(list_task[num_task], model, method),
                    'upper_bound')

                if method == 'mean_std_005':
                    mu = 0.9
                    tou_u= 0.95
                    tou_l=1-tou_u
                elif method == 'mean_std_01':
                    mu = 0.8
                    tou_u= 0.9
                    tou_l=1-tou_u

                elif method == 'mean_std_015':
                    mu = 0.7
                    tou_u= 0.85
                    tou_l=1-tou_u

                elif method == 'mean_std_02':
                    mu = 0.6
                    tou_u= 0.80
                    tou_l=1-tou_u

                elif method == 'mean_std_025':
                    mu = 0.5
                    tou_u= 0.75
                    tou_l=1-tou_u

                elif method == 'QR_005':
                    mu = 0.9
                    tou_u= 0.95
                    tou_l=1-tou_u

                elif method == 'QR_01':
                    mu = 0.8
                    tou_u= 0.9
                    tou_l=1-tou_u

                elif method == 'QR_015':
                    mu = 0.7
                    tou_u= 0.85
                    tou_l=1-tou_u

                elif method == 'QR_020':
                    mu = 0.6
                    tou_u= 0.8
                    tou_l=1-tou_u

                elif method == 'QR_025':
                    mu=0.5
                    tou_u= 0.75
                    tou_l=1-tou_u

                PI_df_02, PICP, MPIW, NPIW, CWC= cal_PI(upper_bound, lower_bound, actual_price, r, mu)
                list_pinball = []
                for UB, LB, AC in zip(upper_bound, lower_bound, actual_price):
                    # if AC < LB:
                    #     pinball = (LB - AC) * (1 - tou)
                    # elif AC > UB:
                    #     pinball = (AC - UB) * tou
                    # else:
                    #     if AC > (UB + LB) / 2:
                    #         pinball = (AC - (UB + LB) / 2) * tou
                    #     else:
                    #         pinball = ((UB + LB) / 2 - AC) * (1 - tou)
                    #     pinball = 0

                    if AC < LB:
                        pinball_1 = (LB - AC) * (1 - tou_l)
                        pinball_2 = (UB - AC) * (1 - tou_u)
                    elif AC > UB:
                        pinball_1 = (AC - LB) * tou_l
                        pinball_2 = (AC - UB) * tou_u
                    else:
                        pinball_1= (AC-LB)*tou_l
                        pinball_2= (UB-AC)*(1-tou_u)
                        pinball_1 = 0
                        pinball_2 = 0

                    # print(pinball)
                    pinball = (pinball_1 + pinball_2) / 2
                    list_pinball.append(pinball)
                avg_pinball_val=np.array(list_pinball).mean()
                print(avg_pinball_val)
                firebase.put('GEFcom2014_spike/{}/results/model-{}/{}'.format(task, model, method), 'PICP', PICP)
                firebase.put('GEFcom2014_spike/{}/results/model-{}/{}'.format(task, model, method), 'MPIW', MPIW)
                firebase.put('GEFcom2014_spike/{}/results/model-{}/{}'.format(task, model, method), 'NPIW', NPIW)
                firebase.put('GEFcom2014_spike/{}/results/model-{}/{}'.format(task, model, method), 'CWC', CWC)
                firebase.put('GEFcom2014_spike/{}/results/model-{}/{}'.format(task, model, method), 'avg_pinball',
                             avg_pinball_val)

                # calculate pinball loss function

    print('finish {}'.format(task))
print('finish')




