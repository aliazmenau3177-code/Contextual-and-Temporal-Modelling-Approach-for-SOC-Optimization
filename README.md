# Title: Contextual and Temporal modeeling Approach for SOC optimization through ML, DM and Big Data Analysis
# Description
This python code is a result of our research in optimizing Security Operations Centre through a combination of Big Data Analysis incorporating data mining and machine learning techniques. 
The code is a practical execution of gradient boosted classfiier comibined with execution of graph stream and sequence stream mechanisms, to enable SOC to optimiize and reduce false postivies by obtaining contextual and temporal awareness of the cyber activity wtihin its responsibility. Moreover through thorough ablation testing we have also discovered that the common perception of combining several ML classifiers with anomaly detection ensembles to optimize SOC and claim notabel reduction in false positives, is a misconception.
The reason is that such a mechanism which has been widely adopted by previous researchers, actually makes SOC memorize the events/ offences and their sources rather than enabling it to think basing on true contextual and temporal awareness. 
The execution of our code and its results (generated within the same script) prove otherwise. 
The relevant research for this code has been written, is aimed at laying the foundation stone for achieveing a fully automated, self driven contextually aware SOC one day.

# Dataset Information 
The dataset utilized for executing the code upon is Microsoft's GUIDE dataset, the SOC specific dataset released in 2025 for public research. 
The original dataset of Microsoft GUIDE consisted of raw telemetry data related to SOC, consisting of 46 features/ columns, which can be divided into 5 main cateogries of information as follows: -
1. Identifiers
2. Alert Meta Data
3. MITRE ATT&CK annotations
4. Host/ network context
5. Enrichment Labels

The original dataset consists of 7,95,426 rows of records. 
Four sequential processes have been executed on the given dataset to achieve following: -
1. Removal of attributes that do not affect alert correctness or contribute to inferring the type of an offence (i.e. TP/FP)
2. Removal of leakage attirbutes - i.e attributes which may cause a bias within the learning process of the applied ML algorithm
3. Removal of redundancy and noise through feature engineering
4. Normalization and binarization of values
5. Label trasnformation
6. Encoding of important features for utilization by ML algorithms

The resulting code contains the same number of records/ rows as the original one but after the execution of feature engineering and encoding processes the new dataset contains 790 columns

The steps carried out are explained under the heading of methodology

# Code Information 
The code file provided within this repository is named "Script4_attribution.py". it s python file that can be run within any python related IDE.
However we have utilized pycharm for the said purpose. The code contains following execution steps
1. importing of libraries
2. loading of data
3. creating binary levels
4. removing leakage columns from the dataset
5. feature engineering steps
6. splitting features and labels
7. categorical encoding of features
8. export of preprocessed dataset (optional)
9. train-test split
10. execution of random forest algorithm
11. execution of XGboost algorithm
12. execution of isolation forest for anomaly detection
13. fusion of models
14. evaluation of results
15. identifying important features
16. generating graphs of confusion matrix and feature importance
17. execution of shuffle sanity test
    
# Method of Access, Usage Instructions and Loading of Data
The subject dataset i.e Microsoft GUIDE can be downloaded from the link https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction. 
the said dataset is large in size and its utilization within the code depends on your processing resources. you can also take a selected well balanced subset of entire Microsoft GUIDE to execute and test our alogrithms. A sample subset used by us is available of this link. https://drive.google.com/file/d/1NTa6pRXkCLhJL3oW_tn-Mq2u2yNfHZNo/view?usp=sharing 
anyone can access this link and download the zipfile containing a well balanced subset of Microsoft GUIDE dataset, suitable enough for utilizing within our python code. 
## Method of Access - Dataset Link
The sample subset used by our team is available on the link https://drive.google.com/file/d/1NTa6pRXkCLhJL3oW_tn-Mq2u2yNfHZNo/view?usp=sharing 
## Loading of Data
Any researcher who desires to utilize this  dataset version of Microsoft GUIDE, may download it to his specific folder, from where he can upload it into any of his python program by adding the following command within his script :-

DATA_PATH = "/(path to the folder location where the downloaded dataset is stored)/processed_GUIDE.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)
print(df['IncidentGrade'].value_counts())

## Usage Instructions
once the dataset has been placed in the desired folder and correct path has been specified within the given code, the code can be executed within the IDE environment. In our case we have executed it within Pycharm. the code will automatically carry out all steps specified within the code information above and provide you with results. 
# Requirements
In order to upload and utilize the data within python scripts, the specific python libraries requried can be called wtihin the script as follows: -

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")

the processing/ computation requirement for running this code are as follows:-
1. one processor machine with at least 4 cores
2. minimum 16 GB of RAM
3. pycharm (installed on UBUNTU)

# Methodology 
The processes executed on the original Microsfot GUIDE Dataset to achieve its pre-processed, Machine Learning ready version, are enumerated as follows: -

1. Column normailzation after dataset loading
2. Binary Label construction
3. Leakage attirbutes removal
4. Irrelevant attributes removal
5. Feature engineering
   a. Temporal feature engineering - transforming raw timestamp into time of day & day of week format in order to be utilized within the ML process
   b. MITRE feature engineering - multi valued MITRE ATT&CK technique string converted into numerical indicators basing on counts of various techniques occuring.
6. One hot encoding - converting all categorical variables into binary numerical vectors

7. after achieving the pre-processed version of GUIDE dataset the algorithms speicifed within out research paper are executed on it. the results alongwith analytical graphs are shown through the same code. 


