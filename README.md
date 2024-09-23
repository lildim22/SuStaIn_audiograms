# SuStaIn_audiograms
Code repository for application of SuStaIn to audiogram data 

# Project Overview: 


# File Descriptions: 
nahnes_all_leftEar+ld.csv: Data file used for pilot
nhanes_v2.sh : Shell script that automates the execution of the project's Python scripts in sequence
nhanes_1000_run.py: Runs SuStaIn on 1000 audiograms records

Example: data_preprocessing.py: Cleans and preprocesses the input data.
# Installation Instructions: Any dependencies or setup required.
# Usage Guide: Explain the order in which to run the scripts or notebooks.

In terminal run ./nhanes_v2.sh. 
Ensure the script has execute permissions chmod +x nhanes_v2.sh

Run data_preprocessing.py to clean the data.
Run model_training.py to train the model.
Use model_evaluation.py to validate the model on test data.
Expected Outputs: Briefly mention what outputs to expect after running the scripts.
