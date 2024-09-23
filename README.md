# SuStaIn_audiograms
Code repository for application of SuStaIn to audiogram data 

# Title: Phenotyping Sensorineural Hearing Loss using SuStaIn    

# Table of contents
- Introduction
- Installation
- License 
- Contact 

# Introduction
This project aims to identify sensorineural hearing loss subtypes with distinct progression patterns using the Subtype and Staging Inference (SuStaIn). 

Background
The majority of hearing loss is sensorineural in origin. Whilst we have gained insights into the underlying causes and mechanisns of sensorineural hearing loss in the lab, we do not have the tools to identify the underlying cause of hearing loss in patients. This means we cannot identify which patients in this diverse cohort will benefit from new, targeted treatments for hearing loss.

SuStaIn is an algorithm for discovery of data-driven groups or "subtypes" in chronic disorders. It is advantageous over other algorithms because it takes into account that individuals can belong to different disease subtypes but also that these disease subtypes are dynamic and may exist across a number of stages. 

Progression is modelled as a series of ordered, discretised events independent of time scale where each event describes a change in a biomarker (in this case a biomarker is the hearing threshold at 1 of 12 test frequencies performed in the PTA) by transforming raw values of the biomarker of the pathological group (patients with hearing loss) against the healthy control population to calculate a z-score. A linear z-score model will be used to define progression as a linear trajectory between one z-score to the next. Further information can be found: https://github.com/ucl-pond/pySuStaIn/blob/master/README.md

# Installation and dependencies

Create new environment and install SuStaIn and dependencies as per https://github.com/ucl-pond/pySuStaIn/blob/master/notebooks/SuStaInWorkshop.ipynb
Steps repeated here below:

Step 1: Open up a terminal window and create a new environment "sustain_env" in anaconda that uses python 3.7 and activate the environment ready to install pySuStaIn.  
conda create --name sustain_tutorial_env python=3.7
conda activate sustain_tutorial_env

Step 2: Use the terminal to install necessary packages for running the notebook and pySuStaIn within the environment.  
conda install -y ipython jupyter matplotlib statsmodels numpy pandas scipy seaborn pip
pip install git+https://github.com/ucl-pond/pySuStaIn

# License 
This project is licensed under the MIT License - see the LICENSE file for details.

# Contact
Created by Lilia Dimitrov and Liam Barrett - feel free to contact us at l.dimitrov@ucl.ac.uk and l.barrett.16@ucl.ac.uk

