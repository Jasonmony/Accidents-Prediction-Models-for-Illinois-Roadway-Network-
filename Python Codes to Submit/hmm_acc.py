#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 19:15:14 2021

@author: jacob
"""
import numpy as np
from HMM_Course import HMM


# Single state condition
def get_prediction(A, a = 0.25, b = 0.5):
    predictions = []
    for i in range(len(TestingData)):
        # res = lm.predict_proba(X_Test_Scaled[i].reshape(1, -1))[0]
        # res_high = np.array([res[0]*(1-b),res[1]*(1-a), res[2]*(1+a), res[3]*(1+b)])
        # res_low = np.array([res[0]*(1+b),res[1]*(1+a), res[2]*(1-a), res[3]*(1-b)])
        # res_high_norm = res_high/res_high.sum()
        # res_low_norm = res_low/res_low.sum()
        # B = np.vstack((res_high_norm, res_low_norm))
        res = lm.predict_proba(X_Test_Scaled[i].reshape(1, -1))
        B = res
        
        seq = TestingData[['Obs_06','Obs_07','Obs_08','Obs_09']].iloc[i].to_list()
        
        model = HMM(A, B, 
                states=['High'],
                emissions=lm.classes_.tolist())
        if all(isinstance(i, str) for i in seq):
            seq_num = [model.emissions.index(i) for i in seq]

        Alpha = model.forward_backward(seq_num)
        
        # print(seq, TestingData['Observation'].iloc[i], Alpha[-1,:]@ model.A @ model.B)
        
        ind = np.argmax(Alpha[-1,:]@ model.A @ model.B)
        predictions.append(model.emissions[ind])
        # if i == 10:
            # break
        # print(seq, model.emissions[ind])
    return predictions


A1 = np.array([[0.50, 0.50],
              [0.50, 0.50]])
A2 = np.array([[0.80, 0.20],
              [0.90, 0.10]])
A3 = np.array([[1]]) 

predictions0 = get_prediction(A = A3, a = 0, b = 0)


from sklearn.metrics import accuracy_score
accuracy_score(TestingData['Observation'], predictions0)



# model = HMM(A, B, 
#             states=['High', 'Low'],
#             emissions=['Zero','Low','Med','High'])

# Alpha = model.forward_algorithm(seq)
# Beta = model.backward_algorithm(seq)
# res = model.forward_backward(seq)
# np.argmax(Alpha[-1,:]@ model.A @ model.B)
            
            
            
            
def get_prediction_2(A, a = 0.25, b = 0.5): #using 3 category logistic
    predictions = []
    for i in range(len(TestingData)):
        res = lm.predict_proba(X_Test_Scaled[i].reshape(1, -1))[0]
        res_high = np.array([res[0]*(1-b),res[1]*(1-a), res[2]*(1+a), res[3]*(1+b)])
        res_low = np.array([res[0]*(1+b),res[1]*(1+a), res[2]*(1-a), res[3]*(1-b)])
        res_high_norm = res_high/res_high.sum()
        res_low_norm = res_low/res_low.sum()
        B = np.vstack((res_high_norm, res_low_norm))
        
        seq = TestingData[['Obs_06','Obs_07','Obs_08','Obs_09']].iloc[i].to_list()
        
        model = HMM(A, B, 
                states=['High', 'Low'],
                emissions=lm.classes_.tolist())
        if all(isinstance(i, str) for i in seq):
            seq_num = [model.emissions.index(i) for i in seq]
            
        Alpha = model.forward_algorithm(seq_num)

        ind = np.argmax(Alpha[-1,:]@ model.A @ model.B)
        predictions.append(model.emissions[ind])
        # print(seq, model.emissions[ind])
    return predictions

predictions_v2_0 = get_prediction_2(A = A1, a = 0, b = 0)
accuracy_score(TestingData['Observation'], predictions_v2_0)
predictions_v2_1 = get_prediction_2(A = A1, a = 0.25, b = 0.5)
accuracy_score(TestingData['Observation'], predictions_v2_1)
predictions_v2_2 = get_prediction_2(A = A1, a = 0.1, b = 0.2)
accuracy_score(TestingData['Observation'], predictions_v2_1)

predictions_v3_0 = get_prediction_2(A = A2, a = 0, b = 0)
accuracy_score(TestingData['Observation'], predictions_v3_0)
predictions_v3_1 = get_prediction_2(A = A2, a = 0.25, b = 0.5)
accuracy_score(TestingData['Observation'], predictions_v3_1)
predictions_v3_2 = get_prediction_2(A = A2, a = 0.1, b = 0.2)
accuracy_score(TestingData['Observation'], predictions_v3_2)