# SuStaIn_audiograms
Code repository for application of SuStaIn to audiogram data from NHANES in pilot study 

# Project Overview: 
This project involves a pilot study to evaluate the feasibility, computational time, and resource demands of running the SuStaIn (Subtype and Stage Inference) model on audiometric data. The primary goal is to identify potential performance challenges and adjust model parameters before scaling up to a full analysis.

We utilized publicly available audiometric data from the National Health and Nutrition Examination Survey (NHANES). NHANES is a program designed to assess the health of adults and children in the United States through cross-sectional evaluations of representative participants. Audiometric data from the NHANES audiometry component for the years 1999–2012 and 2015–2020 were downloaded and merged into a single dataset. The dataset contains air-conduction (AC) measures across five test frequencies: 0.5, 1, 2, 4 and 8 kHz for the left ear per participant.

# File Descriptions: 
nahnes_all_leftEar+ld.csv: Data file used for pilot

sustain_output_full.pkl: Pickle file containing outputs from 1st python script needed in 2nd and 3rd scripts 

nhanes_v2.sh : Shell script that automates the execution of the project's Python scripts in sequence (_run, _mcmc_trace and _pvd)

nhanes_1000_run.py: Runs SuStaIn on 1000 audiograms records

nhanes_1000_mcmc_trace.py: Generates line plot and histogram of log likelihood across MCMC samples

nhanes_pvd_new.py: Plots positional variance diagram to interpret the subtype progression pattern  

nhanes_get_subtype.py: Assigns each record to a subtype and stage. 

nhanes_zdata_with_subtypes.csv: Z-score transformed dataset - needed as input for jupyter notebook below

nhanes_subtype_characteristics.ipynb: Jupyter notebook to evaluate characteristics of the subtypes in the optimal model 

# Installation Instructions: Any dependencies or setup required.
Create new environment and install SuStaIn and dependencies as per https://github.com/ucl-pond/pySuStaIn/blob/master/notebooks/SuStaInWorkshop.ipynb
Steps repeated here below:

Step 1: Open up a terminal window and create a new environment "sustain_env" in anaconda that uses python 3.7 and activate the environment ready to install pySuStaIn.

conda create --name sustain_tutorial_env python=3.7
conda activate sustain_tutorial_env

Step 2: Use the terminal to install necessary packages for running the notebook and pySuStaIn within the environment.

conda install -y ipython jupyter matplotlib statsmodels numpy pandas scipy seaborn pip
pip install git+https://github.com/ucl-pond/pySuStaIn

# Usage Guide: Explain the order in which to run the scripts or notebooks.

Step 1: 
In terminal run ./nhanes_v2.sh. 
Ensure the script has execute permissions chmod +x nhanes_v2.sh
Ensure the data file in same directory as the shell and python scripts 

Step 2: 
Run nhanes_1000_mcmc_trace.py to plot the MCMC trace and histogram
Run nhanes_pvd_new.py to plot the positional-variance diagram # ensure to specify the optimal subtype number for plotting 

Step 3: 
Run nhanes_get_subtype.py to get the df with z-scores and subtype and stage allocation for each record
Upload the above output as csv file (nhanes_zdata_with_subtypes.csv) into jupyter notebook (nhanes_subtype_characteristics.ipynb) for analysis of the characteristics of the subtypes 


