import pandas as pd
import numpy as np
import neurolab as nl
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"]=10.0
plt.rcParams['figure.figsize'] = 8, 5


list_task=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]


x=list(range(0,24,1))
df1=pd.read_csv('actual_price.csv', sep=':')



df_benchmark=pd.DataFrame()

#get benchmark
for num_task in list_task:
    df_temp=pd.read_csv('GEFCom2014_Data/Price/Task {}/Benchmark{}_P.csv'.format(num_task+1, num_task+1))
    list_benchmark=list(df_temp['0.01'])
    df_benchmark['task_{}'.format(num_task+1)]=list_benchmark



df2=pd.read_csv('price_spike_model2_QR_010_LB.csv', sep=':')
df3=pd.read_csv('price_spike_model2_QR_010_UB.csv', sep=':')





plt.figure(1)
for num_task in list_task[3:]:
    print(num_task)
    plt.subplot(4,3,num_task-2)
    plt.plot(x, df1.iloc[num_task,1:25], label="real_price")
    plt.fill_between(x, df2.iloc[num_task,1:25], df3.iloc[num_task,1:25], label='QR_01', linestyle='--', color='red', alpha=0.5)
    plt.title('h{}'.format(num_task+1))
    plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.35)
plt.legend(loc='br')
plt.show()





df2=pd.read_csv('price_spike_model2_MV_010_LB.csv', sep=':')
df3=pd.read_csv('price_spike_model2_MV_010_UB.csv', sep=':')

plt.figure(2)
for num_task in list_task[3:]:
    print(num_task)
    plt.subplot(4,3,num_task-2)
    plt.plot(x, df1.iloc[num_task,1:25], label="real_price")
    plt.fill_between(x, df2.iloc[num_task,1:25], df3.iloc[num_task,1:25], label='QR_01', linestyle='--', color='green', alpha=0.5)
    plt.title('h{}'.format(num_task+1))
    plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.35)
plt.legend(loc='br')
plt.show()

#
# df2=pd.read_csv('price_spike_model2_QR_005_LB.csv', sep=':')
# df3=pd.read_csv('price_spike_model2_QR_005_UB.csv', sep=':')
#
# df4=pd.read_csv('price_model2_QR_005_LB.csv', sep=':')
# df5=pd.read_csv('price_model2_QR_005_UB.csv', sep=':')
#
# for num_task in list_task:
#     plt.figure(num_task+3)
#     plt.plot(x, df1.iloc[num_task,1:25], label="real_price", color='black')
#     plt.plot(x,df_benchmark['task_{}'.format(num_task+1)], label="benchmark", color='black',linestyle='--')
#     plt.fill_between(x, df2.iloc[num_task,1:25], df3.iloc[num_task,1:25], label='Price-spike', linestyle='--', linewidth=2, color='grey', alpha=0.3, hatch='//')
#     plt.fill_between(x, df4.iloc[num_task,1:25], df5.iloc[num_task,1:25], label='Non-price-spike', linestyle='-', linewidth=2, color='slategrey', alpha=0.3)
#     plt.title('task {}'.format(num_task+1))
#     plt.legend(loc='br')
#     plt.grid(True)
#     plt.savefig('myfig_{}'.format(num_task+1))
#
#     # plt.show()



df2=pd.read_csv('price_spike_model2_QR_005_LB.csv', sep=':')
df3=pd.read_csv('price_spike_model2_QR_005_UB.csv', sep=':')

df4=pd.read_csv('price_spike_model2_QR_025_LB.csv', sep=':')
df5=pd.read_csv('price_spike_model2_QR_025_UB.csv', sep=':')

for num_task in list_task:
    plt.figure(num_task+3)
    plt.plot(x, df1.iloc[num_task,1:25], label="real_price", color='black')
    plt.plot(x,df_benchmark['task_{}'.format(num_task+1)], label="benchmark", color='black',linestyle='--')
    plt.fill_between(x, df2.iloc[num_task,1:25], df3.iloc[num_task,1:25], label='Price-spike', linestyle='--', linewidth=2, color='grey', alpha=0.3, hatch='//')
    plt.fill_between(x, df4.iloc[num_task,1:25], df5.iloc[num_task,1:25], label='Non-price-spike', linestyle='-', linewidth=2, color='slategrey', alpha=0.3)
    plt.title('task {}'.format(num_task+1))
    plt.legend(loc='br')
    plt.grid(True)
    # plt.savefig('myfig_{}'.format(num_task+1))

    plt.show()

print('finish')





# from matplotlib.ticker import NullFormatter  # useful for `logit` scale
#
# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
# # make up some data in the interval ]0, 1[
# y = np.random.normal(loc=0.5, scale=0.4, size=1000)
# y = y[(y > 0) & (y < 1)]
# y.sort()
# x = np.arange(len(y))
#
# # plot with various axes scales
# plt.figure(1)
#
# # linear
# plt.subplot(221)
# plt.plot(x, y)
# plt.yscale('linear')
# plt.title('linear')
# plt.grid(True)
#
#
# # log
# plt.subplot(222)
# plt.plot(x, y)
# plt.yscale('log')
# plt.title('log')
# plt.grid(True)
#
#
# # symmetric log
# plt.subplot(223)
# plt.plot(x, y - y.mean())
# plt.yscale('symlog', linthreshy=0.01)
# plt.title('symlog')
# plt.grid(True)
#
# # logit
# plt.subplot(224)
# plt.plot(x, y)
# plt.yscale('logit')
# plt.title('logit')
# plt.grid(True)
# # Format the minor tick labels of the y-axis into empty strings with
# # `NullFormatter`, to avoid cumbering the axis with too many labels.
# plt.gca().yaxis.set_minor_formatter(NullFormatter())
# # Adjust the subplot layout, because the logit one may take more space
# # than usual, due to y-tick labels like "1 - 10^{-3}"
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
#                     wspace=0.35)
#
# plt.show()

