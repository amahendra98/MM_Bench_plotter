import os
import numpy as np

DL = ('cINN','INN','MDN','VAE')
dataset = ('Chen','Peurifoy','Yang')
ids = dict(zip(dataset,(2,1,4))) #This is the sample selected from the dataset

# Extracts from raw data files
'''for invs in os.listdir('.'):
    if invs in DL:
        for set in dataset:
            with open('./my-plot-data/'+set+'/Ypred-'+invs+'-'+set+'-'+str(ids[set])+'.csv', 'w') as write_file:
                mse_arr = np.genfromtxt(invs+'/'+set+'/mse_mat.csv',delimiter=' ')
                sample_mse_arr = mse_arr[:,ids[set]]
                print(sample_mse_arr, sample_mse_arr.shape)
                min_T = np.argmin(sample_mse_arr)
                best_T_run = np.genfromtxt(invs+'/'+set+'/test_Ypred_'+set+'_best_modelinference{}.csv'.format(min_T), delimiter=' ')
                y = best_T_run[ids[set],:]
                y = y.reshape((1,len(y)))
                y = np.tile(y,(5,1))
                np.savetxt(write_file,y,delimiter=' ')'''

# Modifies NA data to create file of same fromat ^ for sample id
for set in dataset:
    mat = np.genfromtxt('my-plot-data/'+set+'/Ypred-NA-'+set+'-{}-T200.csv'.format(ids[set]),delimiter=' ')
    truth = np.genfromtxt('my-plot-data/'+set + '/Ytruth.csv', delimiter=' ')[ids[set],:]
    mse_list = np.empty(200)
    for i,spec in enumerate(mat):
        mse_list[i] = np.mean(np.power((truth-spec),2))

    min_T = np.argmin(mse_list)
    best_T_run = mat[min_T,:]

    with open('./my-plot-data/' + set + '/Ypred-NA-' + set + '-' + str(ids[set]) + '.csv', 'w') as write_file:
        y = best_T_run
        y = y.reshape((1, len(y)))
        y = np.tile(y, (5, 1))
        np.savetxt(write_file, y, delimiter=' ')