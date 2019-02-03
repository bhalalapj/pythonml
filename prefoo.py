# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 13:41:12 2019

@author: Bhalala
"""

import numpy as np
from sklearn import preprocessing

input_data = np.array(
    [[3, -1.5, 3, -6.4], [0, 3, -1.3, 4.1], [1, 2.3, -2.9, 4.3]])

data_standardized = preprocessing.scale(input_data)
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(input_data)
data_normalized = preprocessing.normalize(input_data, norm='l1')

print("\nMean =", data_standardized.mean(axis=0))
print("Std Deviation =", data_standardized.std(axis=0))
print("\n Min max scaled data =", data_scaled)
print("L1 normalized data =", data_normalized)
