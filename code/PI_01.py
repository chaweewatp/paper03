from random import sample
import neurolab as nl
import pandas as pd
import numpy as np



def get_train_data(df, POI, hist_hour, area):
    # t-1 h price
    # t-2 h price
    # t-3 h price
    # t-24 h price   (yesterday load)
    # t-48 h price   (last two days load)
    # t-168 h price  (last week load)
    # t-336 h price  (last two week load)
    # hour of day  (0-23)
    # day of week  (0-6)
    # month of year (0-11)
    input_data=np.array([])
    target_data=np.array([])
    for item in POI:
        if item < 336:
            item = item + 336
        for item2 in hist_hour:
            input_data = np.append(input_data, [df[area][item+item2]])
        input_data=np.append(input_data, (float(df.index[item].split(' ')[3]))%24) #hour of day
        input_data = np.append(input_data, (float((df.index[item].split(' ')[1]))%168)//24) #day of week
    # target = np.asarray(df['LoadR{}RT'.format(area)][rindex])
    # target=df[area][POI].values
    # target=[df[area][POI].values,np.mean(df[area][POI]),np.std(df[area][POI]),np.min(df[area][POI]),np.max(df[area][POI])]

        # target_data = np.append(np.mean(df[area][POI]), np.std(df[area][POI], np.min(df[area][POI]), np.max(df[area][POI]))])

        # target_data = np.append(target_data, df[area][item])
        # target_data = np.append(target_data, np.mean(df[area][POI]))
        # target_data = np.append(target_data, np.var(df[area][POI]))
        # target_data = np.append(target_data, np.min(df[area][POI]))
        # target_data = np.append(target_data, np.max(df[area][POI]))
        target_data = np.append(target_data, np.mean(df[area][list(np.where(POI%24==item%24)[0])]))
        target_data = np.append(target_data, np.var(df[area][list(np.where(POI%24==item%24)[0])]))
        target_data = np.append(target_data, np.min(df[area][list(np.where(POI%24==item%24)[0])]))
        target_data = np.append(target_data, np.max(df[area][list(np.where(POI%24==item%24)[0])]))


    # target_mean=np.mean(df[area][POI])
    # target_std=np.std(df[area][POI])
    # target_mix=np.min(df[area][POI])
    # traget_max=np.max(df[area][POI])
    # print(type(target))
    return input_data.reshape(len(POI), len(hist_hour)+2), target_data.reshape(len(POI), 4)




df1 = pd.read_pickle('res_all_year_THP.pkl')

max_df=df1.max(axis=1)+1
min_df=df1.min(axis=1)-1
df1 = df1.transpose()


#batch_1
POI=np.arange(2000,3050,1)
area =1
hist_hour=[-1,-2,-3,-24,-48,-168,-336]

input_data, target_data = get_train_data((df1-min_df[area])/(max_df[area]-min_df[area]), POI, hist_hour, area)


net_area= nl.net.newff([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,23],[0,6]],[9,4])
err = net_area.train(input_data, target_data, show=15)

#sim
input_data, target_data = get_train_data((df1 - min_df[area]) / (max_df[area] - min_df[area]), POI,
                                         hist_hour, area)
temp_output = ((net_area.sim(input_data)) * (max_df[area] - min_df[area])) + min_df[area]
print(temp_output)
print('finish')
