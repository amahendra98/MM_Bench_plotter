import numpy as np
import os

for d in ['Peurifoy', 'Yang_sim', 'Chen']:
    dat_dir = '../multi-eval-dat/NA/'+d
    avg_std = 0
    count = 0
    for f in os.listdir(dat_dir):
        if 'inference' in f and 'Xpred' in f:
            count += 1
            mat = np.genfromtxt(os.path.join(dat_dir,f))
            avg_std += np.std(mat,axis=0)
            #print(count, f,'\t',np.std(mat))

    print("AVERAGE STD DEV: ",np.mean(avg_std/count))

    #NA
    #   Peurifoy AVERAGE STD DEV OF EACH PARAMETER OVER TOP 200 SAMPLES: 0.18526312770182127
    #   Yang_sim AVERAGE STD DEV OF EACH PARAMETER OVER TOP 200 SAMPLES: 0.46208605587578605
    #   Chen AVERAGE STD DEV OF EACH PARAMETER OVER TOP 200 SAMPLES: 0.07157879874013745
