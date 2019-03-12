import pandas as pd
import numpy as np
import neurolab as nl

def get_train_data(inp_df, max_total_load, max_zonal_load, max_zonal_price):
    input = np.asarray([[inp_df.iloc[ind,0]/max_total_load, inp_df.iloc[ind,1]/max_zonal_load ] for ind in inp_df.index])
    target_1 = np.asarray([[inp_df['Zonal Price'].mean()/max_zonal_price, inp_df['Zonal Price'].std()/max_zonal_price] for ind in inp_df.index])
    target_2 = np.asarray([[inp_df['Zonal Price'].quantile(0.05)/max_zonal_price, inp_df['Zonal Price'].quantile(0.95)/max_zonal_price] for ind in inp_df.index])
    return input, target_1, target_2

def get_test_data(inp_df, max_total_load, max_zonal_load):
    output = np.asarray([[inp_df.iloc[ind,0]/max_total_load, inp_df.iloc[ind,1]/max_zonal_load] for ind in inp_df.index])
    return output

def get_sim_data(net, inp, max_zonal_price):
    temp_output = net.sim(inp)
    # print(temp_output)
    output1=[item[0]*max_zonal_price for item in temp_output]
    output2=[item[1]*max_zonal_price for item in temp_output]

    # output1= temp_output[0][0]*max_zonal_price
    # output2 = temp_output[0][1] * max_zonal_price
    return output1, output2

def PI_consutruct_mean_var(var_mean, var_std):
    upper=[var_mean[ind]+1.96*var_std[ind] for ind in list(np.arange(0,len(var_mean),1))]
    lower=[var_mean[ind]-1.96*var_std[ind] for ind in list(np.arange(0,len(var_mean),1))]
    return upper, lower


def PI_construct_quantile(lower_bound, upper_bound):
    upper = upper_bound
    lower = lower_bound
    return lower, upper

def cal_PI(upper, lower, actual_price):
    number_of_price=len(actual_price)
    temp_df=pd.DataFrame()
    temp_df['upper']=upper
    temp_df['lower']=lower
    temp_df['actual price']=actual_price
    temp_df['actual price < upper']=[1 if temp_df.iloc[ind, 0] >= temp_df.iloc[ind, 2] else 0 for ind in list(temp_df.index)]
    temp_df['actual price > lower']=[1 if temp_df.iloc[ind, 1] <= temp_df.iloc[ind, 2] else 0 for ind in list(temp_df.index)]
    temp_df['within boundary']=[1 if temp_df.iloc[ind,3] == temp_df.iloc[ind,4] else 0 for ind in list(temp_df.index)]
    PICP=temp_df['within boundary'].sum()/number_of_price

    temp_df['upper - lower']=temp_df['upper']-temp_df['lower']
    MPIW = temp_df['upper - lower'].sum()/number_of_price
    NPIW = MPIW/(temp_df['actual price'].max()-temp_df['actual price'].min())
    # print(temp_df['actual price'].max()-temp_df['actual price'].min())
    return PICP, MPIW, NPIW
def train_NN(inp_df, input, target):

    net = nl.net.newff([[0,1], [0,1]], [5, 2])
    err = net.train(input, target, epochs=500, show=100, goal=0.02)
    output = net.sim(input)
    return output



df1=pd.read_csv('GEFCom2014_Data/Price/Task 1/Task1_P.csv')
forecast_df = df1[pd.isnull(df1).any(axis=1)].copy()
forecast_df= forecast_df.reset_index(drop=True)


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
set_weekend = set(df1.iloc[list(set(np.where(df1['day_type']==0)[0]).union(set(np.where(df1['day_type']==1)[0]))),5])
df1['holiday']=[1 if date in list(set_holiday.union(set_weekend)) else 0 for date in df1['Date']]

day = 1
hour = 0
df2=df1.drop(list(np.where(df1['day_type']!=day)[0])).copy()  #drop column which day_type not equal to 0 or select only day_type =0
df2=df2.reset_index(drop=True)
df3=df2.drop(list(set(np.where(df2['Hour']!=hour)[0]))).copy() #drop column which day_type not equal to 0 or select only day_type =0
df3=df3.reset_index(drop=True)


# start PI
df1_1=df1.dropna().copy()
df2_1=df2.dropna().copy()

# create input  np.array([[x0, y0,],[x1, y1], ..., [xn, yn]]) # create output  (mean , std) or (LUBE quantile 0.05, 0.95)

input1, target1, target2 = get_train_data(df2_1, df2_1['Forecasted Total Load'].max(), df2_1['Forecasted Zonal Load'].max(), df2_1['Zonal Price'].max())
net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
err = net.train(input1, target1, epochs=500, show=10, goal=0.02)



forecast_df = df1[pd.isnull(df1).any(axis=1)].copy()
forecast_df= forecast_df.reset_index(drop=True)

input2 = get_test_data(forecast_df, df2_1['Forecasted Total Load'].max(), df2_1['Forecasted Zonal Load'].max())
var_mean, var_std = get_sim_data(net, input2, df2_1['Zonal Price'].max())
upper, lower = PI_consutruct_mean_var(var_mean, var_std)
#evaluation period
actual_price=[34.02, 28.26,24.82,22.9,21.59,21.33,21.55,22.49,26.25,28.95,32.17,32.44,34.25,35.97,38.05,37.77,40.87,37.51,35.44,35.28,37.55,38.41,35,32.31]
#PICP, MWIN evaluation
PICP, MPIW, NPIW = cal_PI(upper, lower, actual_price)



#quantile method
input1, target1, target2 = get_train_data(df2_1, df2_1['Forecasted Total Load'].max(), df2_1['Forecasted Zonal Load'].max(), df2_1['Zonal Price'].max())
net = nl.net.newff([[0, 1], [0, 1]], [5, 2])
err = net.train(input1, target2, epochs=500, show=10, goal=0.02)



forecast_df = df1[pd.isnull(df1).any(axis=1)].copy()
forecast_df= forecast_df.reset_index(drop=True)

input2 = get_test_data(forecast_df, df2_1['Forecasted Total Load'].max(), df2_1['Forecasted Zonal Load'].max())
lower_bound, upper_bound = get_sim_data(net, input2, df2_1['Zonal Price'].max())
lower, upper = PI_construct_quantile(lower_bound, upper_bound)
#evaluation period
actual_price=[34.02, 28.26,24.82,22.9,21.59,21.33,21.55,22.49,26.25,28.95,32.17,32.44,34.25,35.97,38.05,37.77,40.87,37.51,35.44,35.28,37.55,38.41,35,32.31]
#PICP, MWIN evaluation
PICP, MPIW, NPIW = cal_PI(upper, lower, actual_price)



print('finish')