import os
import pandas as pd
import pickle
from scipy import stats

# Load variables saved from script 1
with open('sustain_output_full.pkl', 'rb') as f:
    output_folder, dataset_name, N_S_max, N_iterations_MCMC, zdata, sustain_input = pickle.load(f)

zdata_copy = zdata.copy()
# Load pickle file (SuStaIn output) and get the sample log likelihood values
s = 3
pickle_filename_s = os.path.join(output_folder, 'pickle_files', dataset_name + '_subtype' + str(s) + '.pickle')
pk = pd.read_pickle(pickle_filename_s)

for variable in ['ml_subtype', # the assigned subtype
                 'prob_ml_subtype', # the probability of the assigned subtype
                 'ml_stage', # the assigned stage 
                 'prob_ml_stage',]: # the probability of the assigned stage
    
    # add SuStaIn output to dataframe
    zdata_copy.loc[:,variable] = pk[variable] 

# let's also add the probability for each subject of being each subtype
for i in range(s):
    zdata_copy.loc[:,'prob_S%s'%i] = pk['prob_subtype'][:,i]

zdata_copy.to_csv('nhanes_zdata_with_subtypes.csv', index=False)  # Set index=False to avoid writing row numbers to the file