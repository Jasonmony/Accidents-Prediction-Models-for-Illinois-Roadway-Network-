#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 11:59:00 2021

@author: jacob
"""

import pandas as pd
import numpy as np

AllData = pd.read_csv('../all_data_with_spd_limt_v2.csv')

AllData = AllData.drop(['Unnamed: 0'], axis=1)

# Find Segments in the year 2010 which are also present in the year 
AllData_06 = AllData[AllData.Year == 6]
AllData_07 = AllData[AllData.Year == 7]
AllData_08 = AllData[AllData.Year == 8]
AllData_09 = AllData[AllData.Year == 9]
AllData_10 = AllData[AllData.Year == 10]

AllData_06.index = range(len(AllData_06))
AllData_07.index = range(len(AllData_07))
AllData_08.index = range(len(AllData_08))
AllData_09.index = range(len(AllData_09))
AllData_10.index = range(len(AllData_10))

# present_in_all = []
# for i, row in AllData_10.iterrows():
#     check6 = ((AllData_06['cnty_rte'] == row['cnty_rte']) & (AllData_06['begmp'] == row['begmp']) & (AllData_06['endmp'] == row['endmp'])).any()
#     check7 = ((AllData_07['cnty_rte'] == row['cnty_rte']) & (AllData_07['begmp'] == row['begmp']) & (AllData_07['endmp'] == row['endmp'])).any()
#     check8 = ((AllData_08['cnty_rte'] == row['cnty_rte']) & (AllData_08['begmp'] == row['begmp']) & (AllData_08['endmp'] == row['endmp'])).any()
#     check9 = ((AllData_09['cnty_rte'] == row['cnty_rte']) & (AllData_09['begmp'] == row['begmp']) & (AllData_09['endmp'] == row['endmp'])).any()
    
#     present_in_all.append(check6 & check7 & check8 & check9)

# true_present_in_all_ind = [i for i, x in enumerate(present_in_all) if x]

present_in_all_adjusted = []
for i, row in AllData_10.iterrows():
    check6 = ((AllData_06['cnty_rte'] == row['cnty_rte']) & (abs(AllData_06['begmp'] - row['begmp']) <= 0.01) & (abs(AllData_06['endmp'] - row['endmp']) <= 0.01)).any()
    check7 = ((AllData_07['cnty_rte'] == row['cnty_rte']) & (abs(AllData_07['begmp'] - row['begmp']) <= 0.01) & (abs(AllData_07['endmp'] - row['endmp']) <= 0.01)).any()
    check8 = ((AllData_08['cnty_rte'] == row['cnty_rte']) & (abs(AllData_08['begmp'] - row['begmp']) <= 0.01) & (abs(AllData_08['endmp'] - row['endmp']) <= 0.01)).any()
    check9 = ((AllData_09['cnty_rte'] == row['cnty_rte']) & (abs(AllData_09['begmp'] - row['begmp']) <= 0.01) & (abs(AllData_09['endmp'] - row['endmp']) <= 0.01)).any()
    
    present_in_all_adjusted.append(check6 & check7 & check8 & check9)
    # if abs(heads - int(guess)) <= 10:

true_present_in_all_adjusted_ind = [i for i, x in enumerate(present_in_all_adjusted) if x]


# present_in_all_adjusted2 = []
# for i, row in AllData_10.iterrows():
#     check6 = ((AllData_06['cnty_rte'] == row['cnty_rte']) & (abs(AllData_06['begmp'] - row['begmp']) <= 0.05) & (abs(AllData_06['endmp'] - row['endmp']) <= 0.05)).any()
#     check6 = ((AllData_07['cnty_rte'] == row['cnty_rte']) & (abs(AllData_07['begmp'] - row['begmp']) <= 0.05) & (abs(AllData_07['endmp'] - row['endmp']) <= 0.05)).any()
#     check6 = ((AllData_08['cnty_rte'] == row['cnty_rte']) & (abs(AllData_08['begmp'] - row['begmp']) <= 0.05) & (abs(AllData_08['endmp'] - row['endmp']) <= 0.05)).any()
#     check6 = ((AllData_09['cnty_rte'] == row['cnty_rte']) & (abs(AllData_09['begmp'] - row['begmp']) <= 0.05) & (abs(AllData_09['endmp'] - row['endmp']) <= 0.05)).any()
    
#     present_in_all_adjusted2.append(check6 & check7 & check8 & check9)
#     # if abs(heads - int(guess)) <= 10:

# true_present_in_all_adjusted_ind2 = [i for i, x in enumerate(present_in_all_adjusted2) if x]

# present_in_all_adjusted3 = []
# for i, row in AllData_10.iterrows():
#     check6 = ((AllData_06['cnty_rte'] == row['cnty_rte']) & (abs(AllData_06['begmp'] - row['begmp']) <= 0.1) & (abs(AllData_06['endmp'] - row['endmp']) <= 0.1)).any()
#     check6 = ((AllData_07['cnty_rte'] == row['cnty_rte']) & (abs(AllData_07['begmp'] - row['begmp']) <= 0.1) & (abs(AllData_07['endmp'] - row['endmp']) <= 0.1)).any()
#     check6 = ((AllData_08['cnty_rte'] == row['cnty_rte']) & (abs(AllData_08['begmp'] - row['begmp']) <= 0.1) & (abs(AllData_08['endmp'] - row['endmp']) <= 0.1)).any()
#     check6 = ((AllData_09['cnty_rte'] == row['cnty_rte']) & (abs(AllData_09['begmp'] - row['begmp']) <= 0.1) & (abs(AllData_09['endmp'] - row['endmp']) <= 0.1)).any()
    
#     present_in_all_adjusted3.append(check6 & check7 & check8 & check9)
#     # if abs(heads - int(guess)) <= 10:

# true_present_in_all_adjusted_ind3 = [i for i, x in enumerate(present_in_all_adjusted3) if x]


TestingData = AllData_10.iloc[true_present_in_all_adjusted_ind]

AccCount_06 = []
AccCount_07 = []
AccCount_08 = []
AccCount_09 = []
rows_with_issues = []
for i, row in TestingData.iterrows():
    # if i == 2:
        # break
    df6 = AllData_06[(AllData_06['cnty_rte'] == row['cnty_rte']) & (abs(AllData_06['begmp'] - row['begmp']) <= 0.01) & (abs(AllData_06['endmp'] - row['endmp']) <= 0.01)]
    df7 = AllData_07[(AllData_07['cnty_rte'] == row['cnty_rte']) & (abs(AllData_07['begmp'] - row['begmp']) <= 0.01) & (abs(AllData_07['endmp'] - row['endmp']) <= 0.01)]
    df8 = AllData_08[(AllData_08['cnty_rte'] == row['cnty_rte']) & (abs(AllData_08['begmp'] - row['begmp']) <= 0.01) & (abs(AllData_08['endmp'] - row['endmp']) <= 0.01)]
    df9 = AllData_09[(AllData_09['cnty_rte'] == row['cnty_rte']) & (abs(AllData_09['begmp'] - row['begmp']) <= 0.01) & (abs(AllData_09['endmp'] - row['endmp']) <= 0.01)]
    if (len(df6) > 1) or (len(df7) > 1) or (len(df8) > 1) or (len(df9) > 1):
        rows_with_issues.append(i)
        # break
    AccCount_06.append(sum(df6['AccCount']))
    AccCount_07.append(sum(df7['AccCount']))
    AccCount_08.append(sum(df8['AccCount']))
    AccCount_09.append(sum(df9['AccCount']))
    
    # AccCount_06.append()


def get_observation_from_acc_count_list(list_acccount, increased_threshold = 0):
    observation_list = []
    if increased_threshold == 0:
        for i in range(len(list_acccount)):
            if list_acccount[i] == 0:
                observation_list.append('Zero')
            elif list_acccount[i] <= 30:
                observation_list.append('Low')
            elif list_acccount[i] <= 50:
                observation_list.append('Med')
            else:
                observation_list.append('High')
    else:
        for i in range(len(list_acccount)):
            if list_acccount[i] == 0:
                observation_list.append('Zero')
            elif list_acccount[i] <= 10:
                observation_list.append('Low')
            elif list_acccount[i] <= 18:
                observation_list.append('Med')
            else:
                observation_list.append('High')
    return observation_list


observation_06 = get_observation_from_acc_count_list(AccCount_06, 0)
observation_07 = get_observation_from_acc_count_list(AccCount_07, 0)
observation_08 = get_observation_from_acc_count_list(AccCount_08, 0)
observation_09 = get_observation_from_acc_count_list(AccCount_09, 1)

TestingData['Obs_06'] = observation_06
TestingData['Obs_07'] = observation_07
TestingData['Obs_08'] = observation_08
TestingData['Obs_09'] = observation_09

def get_acccount_cat(df):
    if df['Year'] < 9:
        if df['AccCount'] == 0:
            return 'Zero'
        elif df['AccCount'] <= 30:
            return 'Low'
        elif df['AccCount'] <= 50:
            return 'Med'
        else:
            return 'High'
    else:
        if df['AccCount'] == 0:
            return 'Zero'
        elif df['AccCount'] <= 10:
            return 'Low'
        elif df['AccCount'] <= 18:
            return 'Med'
        else:
            return 'High'

# dummy = AllData_06
# dummy['Observation'] = dummy.apply(get_acccount_cat, axis = 1)

AllData_06['Observation'] = AllData_06.apply(get_acccount_cat, axis = 1)
AllData_07['Observation'] = AllData_07.apply(get_acccount_cat, axis = 1)
AllData_08['Observation'] = AllData_08.apply(get_acccount_cat, axis = 1)
AllData_09['Observation'] = AllData_09.apply(get_acccount_cat, axis = 1)

TrainingData = AllData_06.append([AllData_07,AllData_08,AllData_09])
X_Train = TrainingData[['aadt', 'access', 'curv_rad_binary', 'lanewid',
       'med_type_binary', 'no_lanes', 'oneway', 'rodwycls', 'rururb',
       'seg_lng', 'spd_limt', 'surf_cat']]
X_Train['LnAadt'] = np.log(X_Train['aadt'])
X_Train = X_Train.drop(['aadt'], axis=1)


Y_Train = TrainingData['Observation']

TestingData['Observation'] = TestingData.apply(get_acccount_cat, axis = 1)

X_Test = TestingData[['aadt', 'access', 'curv_rad_binary', 'lanewid',
       'med_type_binary', 'no_lanes', 'oneway', 'rodwycls', 'rururb',
       'seg_lng', 'spd_limt', 'surf_cat']]
X_Test['LnAadt'] = np.log(X_Test['aadt'])
X_Test = X_Test.drop(['aadt'], axis=1)


X_Train_Encoded = pd.get_dummies(X_Train,
                     columns = ['access', 'curv_rad_binary', 'med_type_binary', 'oneway','rodwycls','rururb','surf_cat'], drop_first = True)
X_Test_Encoded = pd.get_dummies(X_Test,
                     columns = ['access', 'curv_rad_binary', 'med_type_binary', 'oneway','rodwycls','rururb','surf_cat'], drop_first = True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_Train_Scaled = scaler.fit_transform(X_Train_Encoded)
X_Test_Scaled = scaler.transform(X_Test_Encoded)

from sklearn.linear_model import LogisticRegression

lm = LogisticRegression(multi_class='ovr', solver='liblinear')
lm.fit(X_Train_Scaled, Y_Train)

from sklearn.metrics import accuracy_score
accuracy_score(TestingData['Observation'], lm.predict(X_Test_Scaled))

lm.predict(X_Test_Scaled[0].reshape(1, -1))
lm.predict_proba(X_Test_Scaled[0].reshape(1, -1))


import statsmodels.api as sm



################ Try with Low Medium and High ##################
# TrainingData['Observation_2'] = TrainingData['Observation']
# TrainingData.loc[(TrainingData['Observation'] == 'Zero'),'Observation_2'] = 'Low'
# Y_Train_2 = TrainingData['Observation_2']


# TestingData['Observation_2'] = TestingData['Observation']
# TestingData.loc[(TestingData['Observation'] == 'Zero'),'Observation_2'] = 'Low'
# Y_Test_2 = TestingData['Observation_2']


# TestingData['Obs_06_2'] = TestingData['Obs_06']
# TestingData['Obs_07_2'] = TestingData['Obs_07']
# TestingData['Obs_08_2'] = TestingData['Obs_08']
# TestingData['Obs_09_2'] = TestingData['Obs_09']

# TestingData.loc[(TestingData['Obs_06'] == 'Zero'),'Obs_06_2'] = 'Low'
# TestingData.loc[(TestingData['Obs_07'] == 'Zero'),'Obs_07_2'] = 'Low'
# TestingData.loc[(TestingData['Obs_08'] == 'Zero'),'Obs_08_2'] = 'Low'
# TestingData.loc[(TestingData['Obs_09'] == 'Zero'),'Obs_09_2'] = 'Low'


# lm2 = LogisticRegression(multi_class='ovr', solver='liblinear')
# lm2.fit(X_Train_Scaled, Y_Train_2)

