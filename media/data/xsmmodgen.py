import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.cluster import *
import pickle

dataset = pd.read_csv('final_data.csv')
relevantDataset = dataset.iloc[:, 3:11]
relevantDataset = relevantDataset.drop(['Start Time (s)'], axis = 1)
relevantDataset = relevantDataset.drop(['End Time (s)'], axis = 1)
relevantDataset = relevantDataset.drop(['Duration (s)'], axis = 1)

model = KMeans(n_clusters = 6, random_state = 1).fit(relevantDataset)


pkl_file = "xsm_model.pkl"
with open(pkl_file, 'wb') as file:
    pickle.dump(model, file)




