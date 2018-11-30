
# coding: utf-8

# # Predictive learning

# ## 1. Dataset exploration

# In[1]:


# Imports for the task

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


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

# Obtain X excluding first column
idx_OUT_columns = [0]
idx_IN_columns = [i for i in range(np.shape(npArray)[1]) if i not in idx_OUT_columns]
X = npArray[:,idx_IN_columns]

# Obtain y
y = dataFrame['Free_parking'].values


# In[3]:


# Show first sample data using pandas
dataFrame.head(10)


# <mark>
# <span style="font-size: 16px; color: black">
# Describe the dataset in number of samples, dimensions, classes and samples per class using python.
# </span>
# </mark>

# In[4]:


# Number of samples and dimensions
print("Number of samples, number of features:" + str(X.shape))

# Number of classes
print("Classes:" + str(targetNames))

# Number of samples per class
unique, counts = np.unique(y, return_counts = True)
print("Number of samples per class: " + str(dict(zip(unique, counts))))


# ## 2. Split data into Train, Test

# Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called **overfitting**. To avoid it, it is common practice when performing a supervised machine learning experiment to **hold out part of the available data as a test set XTest, yTest ** and be used only at the end of the data analysis. [[1]](http://scikit-learn.org/stable/modules/cross_validation.html)
# 
# For the splitting CV method ``` StratifiedShuffleSplit ``` (indicating 1 split) has been chosen for this task: it shuffles the dataset assuring the same proportion of samples per class in both Train and Test splits. The sizes chosen for Train-Test splits have been 70%-30% by chosing ```test_size = 0.3``` when calling the method. Additionally, by setting an integer for ```random_state``` attribute we are indicating a seed used by the random number generator for performing the splits with the aim of having the same split from now on in the exercise and compare the operations performed in the different sections in the same way.

# In[5]:


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


# ## 3. Select models

# - **LOGISTIC REGRESSION**
# 
# Despite its name, is a linear model for classification rather than regression. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function. [[2]](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) [[3]](https://en.wikipedia.org/wiki/Logistic_function)
# 
# **C** is the regularization parameter for this model. Parameter **C = 1/λ**. 
# 
# **Lambda (λ)** controls the trade-off between allowing the model to increase it's complexity as much as it wants with trying to keep it simple. For example, if λ is very low or 0, the model will have enough power to increase it's complexity (overfit) by assigning big values to the weights for each parameter. If, in the other hand, we increase the value of λ, the model will tend to underfit, as the model will become too simple.
#     
# Parameter **C** will work the other way around. For small values of C, we increase the regularization strength which will create simple models which underfit the data. For big values of C, we low the power of regularization which imples the model is allowed to increase it's complexity, and therefore, overfit the data. [[4]](https://www.kaggle.com/joparga3/2-tuning-parameters-for-logistic-regression)
# 
# Logistic regression is a classification algorithm traditionally limited to only two-class classification problems. If we have more than two classes then next model, Linear Discriminant Analysis, is the preferred linear classification technique.

# - **LINEAR DISCRIMINANT ANALYSIS**
# 
# A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule. The model fits a Gaussian density to each class, assuming that all classes share the same covariance matrix. The fitted model can also be used to reduce the dimensionality of the input by projecting it to the most discriminative directions. [[5]](http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis)
# 
# LDA consists of statistical properties of the data, calculated for each class. For a single input variable (x) this is the mean and the variance of the variable for each class. For multiple variables, this is the same properties calculated over the multivariate Gaussian, namely the means and the covariance matrix. These statistical properties are estimated from the data and plug into the LDA equation to make predictions.

# - **K NEAREST NEIGHBORS (kNN)**.
# 
# The principle behind nearest neighbor method is to find a predefined number of training samples closest in distance to the new point, and predict the label from these. [[6]](http://scikit-learn.org/stable/modules/neighbors.html#neighbors)
#     
# The **number of neighbors** is the regularization parameter for this model, increasing the model complexity inversely to the number of neighbors. 

# - **DECISION TREE CLASSIFIER**
# 
# Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. [[7]](http://scikit-learn.org/stable/modules/tree.html)
# 
# Decision tree is a classifier in the form of a tree structure. Important parameters are:
# 
# • **Decision node**: specifies a test on a single attribute
# 
# • **Leaf node**: indicates the value of the target attribute
# 
# • **Arc/edge**: split of one attribute
# 
# • **Path**: a disjunction of test to make the final prediction/decision
# 
# Decision trees classify instances or examples by starting at the root of the tree and moving through it until a leaf node.

# - **GAUSSIAN NAIVE BAYES**
# 
# Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of independence between every pair of features. [[8]](http://scikit-learn.org/stable/modules/naive_bayes.html)
# 

# - **SUPPORT VECTOR MACHINES**
# 
# A Support Vector Machine (SVM) is a supervised machine learning algorithm that can be employed for both classification and regression purposes. SVMs are more commonly used in classification problems. SVMs are based on the idea of finding a hyperplane that best divides a dataset into two classes.

# Support vectors are the data points nearest to the hyperplane, the points of a data set that, if removed, would alter the position of the dividing hyperplane. Because of this, they can be considered the critical elements of a data set.
# 
# The distance between the hyperplane and the nearest data point from either set is known as the margin. The goal is to choose a hyperplane with the greatest possible margin between the hyperplane and any point within the training set, giving a greater chance of new data being classified correctly.

# As data is rarely ever clean to classify it easily (see picture above), in order to classify a difficult dataset it’s necessary to move away from a 2d view of the data to a 3d view. This is known as kernelling. As we are now in 3d, our hyperplane can no longer be a line, it must now be a plane as shown in the picture below. The idea is that the data will continue to be mapped into higher and higher dimensions until a hyperplane can be formed to segregate it. [[9]](https://www.kdnuggets.com/2016/07/support-vector-machines-simple-explanation.html)
# 
# Parameters for this estimator are **C** (Penalty parameter of the error term), **gamma** (Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’ kernels) and the **type of kernel**.

# In[6]:


# Array of models
models = []
LR = list(('LOGISTIC REGRESSION', LogisticRegression(random_state=42)),)
LDA = list(('LINEAR DISCRIMINANT ANALYSIS', LinearDiscriminantAnalysis()),)
KNN = list(('K NEAREST NEIGHBORS', KNeighborsClassifier()),)
DT = list(('DECISION TREE', DecisionTreeClassifier(random_state=42)),)
NB = list(('GAUSSIAN NAIVE BAYES', GaussianNB()),)
SVM = list(('SUPPORT VECTOR MACHINES', SVC(random_state=42)),)
MLP = list(('MULTI-LAYER PERCEPTRON', MLPClassifier(random_state=42, max_iter=2000)),)
RF = list(('RANDOM FOREST', RandomForestClassifier(random_state=42)),)


# ## 4. Obtain best estimator hyper-parameters using GridSearch cross-validation
# 
# When evaluating different hyperparameters for the estimator there is still a risk of overfitting on the test set because the parameters can be tweaked until the estimator performs optimally, and consequently knowledge about the test set can “leak” into the model. To solve this problem, another part of the dataset can be held out as a so-called **validation set**: training proceeds on the training set, then evaluation is done on the validation set, and if it is successful, final evaluation can be done on the test set.
# 
# However, by partitioning the available data into three sets, there are two problems: (1) Number of samples which can be used for learning the model is drastically reduced; and (2) the results can depend on a particular random choice for the pair of (train, validation) sets: here it comes **generalization error**, which is a measure of how accurately an algorithm is able to predict outcome values for previously unseen data. 
# 
# A solution to these problems is a procedure called **cross-validation (CV)**. A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. [[13]](http://scikit-learn.org/stable/modules/cross_validation.html). For this task **StratifiedShuffeSplit** has been used.
# 
# The optimal hyperparameters for the models will be obtained using **GridSearch CV** method, which fits every model for every combination of hyper-parameters desired and applies the desired CV over **XTrain, yTrain** sets that we obtained in STEP 2.

# ### Set parameters grid for each model

# In[7]:


#LR
paramGridLR = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
LR.append(paramGridLR)
models.append(LR)


# In[8]:


#LDA
paramGridLDA = {}
LDA.append(paramGridLDA)
models.append(LDA)


# In[9]:


#kNN
paramGridKNN = {
    'n_neighbors': range(1, 30, 2),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}
KNN.append(paramGridKNN)
models.append(KNN)


# In[10]:


#DT
paramGridDT = {
    "max_depth": [3, None],
    "max_features": ['auto', 'sqrt'],
    "min_samples_leaf": range(1, 20, 2),
    "criterion": ["gini", "entropy"]
}

DT.append(paramGridDT)
models.append(DT)


# In[11]:


#NB (no tiene hiperparámetros)
paramGridNB = {}
NB.append(paramGridNB)
models.append(NB)


# In[12]:


#SVM 
paramGridSVM = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['rbf']
}

SVM.append(paramGridSVM)
models.append(SVM)


# In[13]:


#MLP
paramGridMLP = {
    'hidden_layer_sizes': [x for x in itertools.product((10, 30, 50), repeat = 2)],
    'alpha': np.logspace(-5, 3, 5)
}

MLP.append(paramGridMLP)
models.append(MLP)


# In[14]:


#RF
paramGridRF = {'n_estimators': [10, 50],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': range(1, 20, 2)}

RF.append(paramGridRF)
models.append(RF)


# In[15]:


# Show models and hiperparameters used
dfModels = pd.DataFrame(models, columns = ["Model", "Model parameters description", "Parameters Grid"])
pd.set_option('display.max_colwidth', -1)
dfModels


# ### CV using GridSearch

# In[16]:


myCV = StratifiedShuffleSplit(10, 0.3, random_state = 42)

names = []
results = []
bestEstimators = []

for name, model, paramGrid in models:
    
    modelResults = {}
    
    myGridSearchCV = GridSearchCV(model, 
                              paramGrid, 
                              cv = myCV,
                              verbose = 2, 
                              return_train_score = True);
    
    print("MODEL: " + name)
    # Fit the grid
    myGridSearchCV.fit(XTrain, yTrain);
    
    # Scores
    gridScores = pd.DataFrame(myGridSearchCV.cv_results_)
    bestEstimatorResults = gridScores.loc[gridScores['params'] == myGridSearchCV.best_params_]
    
    #names.append(name)
    
    modelResults["Model"] = name
    modelResults["Best estimator parameters"] = (bestEstimatorResults.iloc[0]['params'])
    modelResults["(XTrain,yTrain) Mean test score"] = (bestEstimatorResults.iloc[0]['mean_test_score'])
    modelResults["(XTrain,yTrain) Mean train score"] = (bestEstimatorResults.iloc[0]['mean_train_score'])
    modelResults["(XTrain,yTrain) Std test score"] = (bestEstimatorResults.iloc[0]['std_test_score'])
    modelResults["(XTrain,yTrain) Std train score"] = (bestEstimatorResults.iloc[0]['std_train_score'])
    
    results.append(modelResults)
    
    names.append(name)
    
    bestEstimators.append(myGridSearchCV.best_estimator_)


# In[17]:


# Display GridSearchCV results
results_df_gridsearch = pd.DataFrame(results)
results_df_gridsearch = results_df_gridsearch[['Model', 'Best estimator parameters', 
                                               '(XTrain,yTrain) Mean test score', '(XTrain,yTrain) Std test score', 
                                               '(XTrain,yTrain) Mean train score', '(XTrain,yTrain) Std train score'
                                              ]]

display(results_df_gridsearch)


# Above results shows that all models got a high CV and train score with low standard deviation in CV and trainining. We choose the DecisionTree Model for training the data.
# 
# Let's plot **Validation curve** to analyze under-overfitting considering the complexity of the Decision Tree model in CV.

# ## Validation curve

# In[18]:


# Function for plotting VALIDATION CURVE

def plot_validation_curve(myEstimator, 
                          X, y, 
                          myParamRange,
                          paramName,
                          chartTitle,
                          myCV, 
                          yLower, yUpper):
    
    train_scores, test_scores = validation_curve(myEstimator, 
                                                 X, y, 
                                                 param_name= paramName, 
                                                 param_range = myParamRange,
                                                 cv=myCV, 
                                                 scoring="accuracy", 
                                                 n_jobs=1)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(chartTitle)
    plt.xlabel(paramName)
    plt.ylabel("Accuracy score")
    plt.ylim(yLower, yUpper)
    lw = 2
    plt.semilogx(myParamRange, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(myParamRange, train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(myParamRange, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(myParamRange, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


# *Decision tree*

# In[23]:


plot_validation_curve(bestEstimators[3], 
                      XTrain, yTrain, 
                      paramGridDT['min_samples_leaf'], 
                      'min_samples_leaf',
                      'Validation Curve using DECISION TREES',
                      myCV, 
                      0, 1.05)


# In[29]:


display(results_df_gridsearch["Best estimator parameters"][3])


# ## 5. Fit the Best Models with XTrain, YTrain

# In[38]:


fitModels = []

for model in bestEstimators:
    fitModel = model.fit(XTrain, yTrain)
    
    fitModels.append(fitModel)


# ## 6. Evaluate the score on XTest prediction

# In[39]:


yTrue = yTest

predictions = []

for fitModel in fitModels:
    yPred = fitModel.predict(XTest)
    
    predictions.append(list(yPred))


# In[40]:


#Accuracy scores
accuracy_scores = []

i=0

for prediction in predictions:
    acc_score = accuracy_score(yTrue, prediction)
    accuracy_scores.append(acc_score)
    print(names[i])
    print(acc_score)
    print("\n")
    
    i = i+1

# Append XTest accuracy scores to results
j=0

for result in results:
    result["(XTest) acc score"] = accuracy_scores[j]
    j = j+1

results_df = pd.DataFrame(results)


# ## 7. CV with the complete dataset

# In[41]:


# Cross-validation of the Best Estimator with the entire dataset
myStratifiedShuffleSplit = StratifiedShuffleSplit(100, 0.3, random_state = 42)

i=0

cv_scores = []

for model in fitModels:
    myCrosValScore = cross_val_score(model, X, y, cv = myStratifiedShuffleSplit)
    
    cv_scores.append(myCrosValScore)
    
    print(names[i])
    print ("Mean accuracy score: " + str(np.mean(myCrosValScore)))
    print ("Std deviation: " + str(np.std(myCrosValScore)) + "\n")
    
    i=i+1

# Append accuracy scores to the results
j=0
for result in results:
    result["(X,y) Mean acc score"] = np.mean(cv_scores[j])
    result["(X,y) Std acc score"] = np.std(cv_scores[j])
    j=j+1


# - ### Summary of the results

# In[ ]:


results_df = pd.DataFrame(results)

# Fucntions for stylying scores and std deviations
def highlight_max_score(data, color='lightgreen'):
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)

def highlight_min_std(data, color = 'lightgreen'):
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else:  # from .apply(axis=None)
        is_min = data == data.min().min()
        return pd.DataFrame(np.where(is_min, attr, ''),
                            index=data.index, columns=data.columns)
    
def highlight_min_score(data, color = 'lightsalmon'):
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else:  # from .apply(axis=None)
        is_min = data == data.min().min()
        return pd.DataFrame(np.where(is_min, attr, ''),
                            index=data.index, columns=data.columns)
    
def highlight_max_std(data, color='lightsalmon'):
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)

# Show columns in the desired order
results_df = results_df[['Model', 'Best estimator parameters', 
                         '(XTrain,yTrain) Mean test score', '(XTrain,yTrain) Std test score', 
                         '(XTrain,yTrain) Mean train score', '(XTrain,yTrain) Std train score', 
                         '(XTest) acc score',
                         '(X,y) Mean acc score', '(X,y) Std acc score']]


# Apply styles to the scores
style_max_scores = results_df.style.apply(highlight_max_score, subset=['(XTrain,yTrain) Mean test score',
                                                                       '(XTrain,yTrain) Mean train score',
                                                                       '(XTest) acc score',
                                                                       '(X,y) Mean acc score'
                                                                      ])


style_max_std = results_df.style.apply(highlight_min_std, subset=['(XTrain,yTrain) Std test score',
                                                               '(XTrain,yTrain) Std train score',
                                                               '(X,y) Std acc score'
                                                              ])

style_max_std.use(style_max_scores.export())

style_min_scores = results_df.style.apply(highlight_min_score, subset=['(XTrain,yTrain) Mean test score', 
                                                                '(XTrain,yTrain) Mean train score',
                                                                '(XTest) acc score',
                                                                '(X,y) Mean acc score'
                                                               ])

style_min_scores.use(style_max_std.export())

style_max_std = results_df.style.apply(highlight_max_std, subset=['(XTrain,yTrain) Std test score',
                                                               '(XTrain,yTrain) Std train score',
                                                               '(X,y) Std acc score'
                                                              ])

style_max_std.use(style_min_scores.export()).set_precision(2)


# - ### Learning curves algorithms comparison

# In[87]:


## Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[88]:


i=0
for model in fitModels:
    title = "Learning Curves: " + names[i]

    plot_learning_curve(model, title, X, y, ylim=(0, 1.01), cv=myStratifiedShuffleSplit, n_jobs=4)

    plt.show()
    i=i+1

