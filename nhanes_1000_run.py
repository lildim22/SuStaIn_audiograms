
# Load libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pySuStaIn
import statsmodels.formula.api as smf
from scipy import stats
import sklearn.model_selection

# init sub-dataset label
dataset_fhand = 'nhanes_1000pt_100000MCMC_10st'

#read data
df = pd.read_csv('nahnes_all_leftEar+ld.csv')

data = df.head(1000)
#data columns

biomarkers = data.columns[1:6].tolist()
print("The biomarker columns are:", biomarkers)

#perform normalisation

# make a copy of our dataframe (we don't want to overwrite our original data)
zdata = pd.DataFrame(data,copy=True)

# for each biomarker
for biomarker in biomarkers:
    mod = smf.ols('%s ~ Gender + Age'%biomarker,  # fit a model finding the effect of age and headsize on biomarker
                  data=data[data['HearingLoss']==0] # fit this model *only* to individuals in the control group
                 ).fit() # fit model    
    print(mod.summary())
    
    # get the "predicted" values for all subjects based on the control model parameters
    predicted = mod.predict(data[['Gender','Age',biomarker]]) 

    # calculate our zscore: observed - predicted / SD of the control group residuals
    w_score = (data.loc[:,biomarker] - predicted) / mod.resid.std()
    
    print(np.mean(w_score[data.HearingLoss==0]))
    print(np.std(w_score[data.HearingLoss==0]))
    
    # save zscore back into our new (copied) dataframe
    zdata.loc[:,biomarker] = w_score

## Prepare SuStaIn inputs

N = len(biomarkers)         # number of biomarkers 

SuStaInLabels = biomarkers
Z_max = np.array([14.3, 14.3, 13.7, 14.2, 11.9])
    
Z_vals  = np.array([[ 1.  ,  4.25,  7.5 , 10.75, 14.  ],
                 [ 1.  ,  4.25,  7.5 , 10.75, 14.  ],
                 [ 1.  ,  4.25,  7.5 , 10.75, 14.  ],
                 [ 1.  ,  4.25,  7.5 , 10.75, 14.  ],
                 [ 1.  ,  4.25,  7.5 , 10.75, np.nan  ]]) 

N_startpoints = 25
N_S_max = 10
N_iterations_MCMC = int(1e5)
output_folder = os.path.join(os.getcwd(), '{0}_Output'.format(dataset_fhand))
dataset_name = '{0}_Output'.format(dataset_fhand)


# Initiate the SuStaIn object
sustain_input = pySuStaIn.ZscoreSustain(
                              zdata[biomarkers].values,
                              Z_vals,
                              Z_max,
                              SuStaInLabels,
                              N_startpoints,
                              N_S_max, 
                              N_iterations_MCMC, 
                              output_folder, 
                              dataset_name, 
                              False)

# make the output directory if it's not already created
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
    
import pickle

# Save all necessary variables at the end of script 1
with open('sustain_output_full.pkl', 'wb') as f:
    pickle.dump((output_folder, dataset_name, N_S_max, N_iterations_MCMC, zdata, sustain_input), f)
