import pandas as pd
import numpy as np

def MAE(array1, array2):
    output = sum(abs(array1-array2))/len(array1)
    return output

def MAPE(array1, array2):
    output = 100*(sum(abs((array1-array2)/array1)))/len(array1)
    return output

def Trading_hub_price(bus, res_bus, load_PF):
    THP = [0, 0, 0]
    for item10 in bus.index:
        THP[bus.zone[item10]] = THP[bus.zone[item10]] + res_bus.lam_p[item10] * load_PF[item10]
    return THP


def get_average_hourly_value(df):
    temp_df = df.copy()
    res_df = pd.DataFrame()
    num_of_minutes = 0
    res = np.array([float(0), float(0), float(0)])
    for item in temp_df.columns[:]:
        if num_of_minutes == 0:
            res = np.array([float(0), float(0), float(0)])
        for item2 in np.arange(len(res)):
            res[item2] = res[item2] + temp_df[item][item2]
            # print(res)
        num_of_minutes = num_of_minutes + 1
        if num_of_minutes == 60:
            res_df[item] = res / 60
            num_of_minutes = 0
    return res_df
