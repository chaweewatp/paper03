import pandas as pd
import numpy as np
from firebase import firebase
firebase = firebase.FirebaseApplication('https://thesis-10ad5.firebaseio.com', None)

df1 = pd.read_pickle('res_all_year_THP.pkl')


df2=df1.copy()
df2.columns=list(np.arange(0,len(list(df1)),1))

#construct df for each day
for day in list(np.arange(0,7,1)):
    df2_0=(df2.iloc[:,list(set(np.where((np.asarray(list(df2))// 24)%7 == day)[0]))]).copy()
    df2_0=df2_0.reindex(sorted(df2_0.columns), axis=1)

    #construct df for each day each hour
    for hour in list(np.arange(0,24,1)):
        df2_0_0=(df2_0.iloc[:,list(set(np.where(np.asarray(list(df2_0))% 24 == hour)[0]))]).copy()
        df2_0_0=df2_0_0.reindex(sorted(df2_0_0.columns), axis=1)
        dict_0={area:{day:{hour:{'values':[item for item in df2_0_0.loc[area]], 'quantile':[df2_0_0.loc[area].quantile(quantile) for quantile in [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0]], 'mean':np.mean(df2_0_0.loc[area]), 'std':np.std(df2_0_0.loc[area]), 'max':np.max(df2_0_0.loc[area]), 'min':np.min(df2_0_0.loc[area]), 'lenght': len(df2_0_0.loc[area])}}}for area in list(df2_0_0.index.values)}
        for area in dict_0:
            firebase.put('THP/{}/{}'.format(area,day),'{}'.format(hour), {'data':{'quantile':dict_0[area][day][hour]['quantile'], 'values':dict_0[area][day][hour]['values'], 'mean':dict_0[area][day][hour]['mean'], 'std':dict_0[area][day][hour]['std'], 'min':dict_0[area][day][hour]['min'], 'max':dict_0[area][day][hour]['max'], 'lenght': dict_0[area][day][hour]['lenght']}})



print('finish')


#retieve data
#result = firebase.get('/THP/0/0/0/data', 'max')
