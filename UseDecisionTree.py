
# coding: utf-8

# # PREDICTIVE MODELLING WITH DECISIONTREECLASSIFIER

# In[6]:


from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib


# In[7]:


# Load data
targetNames = [0, 1]
headers = ["Free_parking",  # 0,1 (No/Yes)
           "Id_Parking",       
           "Time_zone",     # 0,1,2,3,4,5,6 (7:00-10:00,10:00-13:00,13:00-16:00,16:00-19:00,19:00-22:00,22:00-00:00,00:00-7:00)
           "Day_of_week",   # 0,1,2,3,4,5,6 (Monday, Tuesday, Wednesday...)
           "Working_day"    # 0,1 (No/Yes)
          ]

dataFrame = pd.read_csv("finalDataset.csv", header = None, sep=',', names = headers)

# Assign data and target to X, y variables to be used later on
npArray = dataFrame.values

# Obtain X exluding the first column
idx_OUT_columns = [0]
idx_IN_columns = [i for i in range(np.shape(npArray)[1]) if i not in idx_OUT_columns]
X = npArray[:,idx_IN_columns]

# Obtain y
y = dataFrame['Free_parking'].values


# In[8]:


## Split the data into Train, Test sets

myStratifiedShuffleSplit = StratifiedShuffleSplit(1, 0.3, random_state = 42)

for train_index, test_index in myStratifiedShuffleSplit.split(X, y):
    XTrain = X[train_index,:]
    XTest = X[test_index,:]
    yTrain = y[train_index]
    yTest = y[test_index]

# Sizes of each data split
print("Number of samples and dimensions for XTrain: " +str(XTrain.shape))
print("Number of labels for yTrain: " +str(yTrain.shape))
print("Number of samples and dimensions for XTest: " +str(XTest.shape))
print("Number of labels for yTest: " +str(yTest.shape))


# In[9]:


##We use DecisionTree method for training
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=1)
clf.fit(XTrain, yTrain)
clf.score(XTest,yTest)


# In[10]:


## The model is saved in a .pkl file 
joblib.dump(clf, 'training_model.pkl')

