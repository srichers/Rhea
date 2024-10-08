'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file is defunct and likely nocontains a variety of functions that generate randomized distributions
'''

# credit to https://thinkingneuron.com/using-artificial-neural-networks-for-regression-in-python/
# Convert the flavor transformation data to one with reduced dimensionality to make it easier to train on
# Run from the directory containin the joint dataset
import h5py
import re
import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer

input_filename = "many_sims_database.h5"
N_Nbar_tolerance = 1e-3

#===============================================#
# read in the database from the previous script #
#===============================================#
f_in = h5py.File(input_filename,"r")
growthRateList     = np.array(f_in["growthRate(1|s)"])
F4_initial_list    = np.array(f_in["F4_initial(1|ccm)"]) # [ind, xyzt, nu/nubar, flavor]
F4_final_list      = np.array(f_in["F4_final(1|ccm)"])
f_in.close()

# N-Nbar must be preserved
ntotal = np.sum(F4_initial_list[:,3,:,:], axis=(1,2))
N_Nbar_initial = F4_initial_list[:,3,0,:] - F4_initial_list[:,3,1,:]
N_Nbar_final   =   F4_final_list[:,3,0,:] -   F4_final_list[:,3,1,:]
N_Nbar_difference = N_Nbar_initial - N_Nbar_final
N_Nbar_error = np.max(np.abs(N_Nbar_difference / ntotal[:,np.newaxis]))
print("N_Nbar_error = ", N_Nbar_error)
assert(N_Nbar_error < N_Nbar_tolerance)

# define the input (X) and output (y) for the neural network
nsims = F4_initial_list.shape[0]
IO_shape = F4_initial_list.shape[1:]
number_predictors = np.product(IO_shape)
X = F4_initial_list.reshape((nsims, number_predictors))
y = F4_final_list.reshape((nsims, number_predictors))

# Standardize the data
PredictorScaler = StandardScaler()
TargetVarScaler = StandardScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)
TargetVarScalerFit=TargetVarScaler.fit(y)

# split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
n_train = X_train.shape[0]
n_test = X_test.shape[0]

# set up sklearn to optimize the hyperparameters

# define custom function to calculate the accuracy
def Accuracy_Score(orig, pred):
    error = np.max(np.abs(orig-pred))
    return 1.0 - error
custom_scoring = make_scorer(Accuracy_Score, greater_is_better=True)

# create test ("Fiducial" simulation)
F4_test = np.zeros(IO_shape)
F4_test[3,0,0] = 1
F4_test[3,1,0] = 1
F4_test[2,0,0] = 1/3
F4_test[2,1,0] = -1/3
F4_test /= np.sum(F4_test[3])


#===============================#
# Abstract out boilerplate code #
#===============================#
def run_ML_model(estimator, param_grid, label):
    print()
    print("######### "+label+" ############")

    grid_search = GridSearchCV(estimator=estimator,
                               param_grid=param_grid,
                               scoring=custom_scoring,
                               cv=5)

    # run the grid search
    StartTime = time.time()
    grid_search.fit(X,y)
    EndTime = time.time()

    print("Total time:",EndTime-StartTime,"seconds.")
    print("Best parameters:",grid_search.best_params_)

    before = F4_test
    after = grid_search.predict(F4_test.reshape((1,number_predictors))).reshape(IO_shape)
    print("N Before:", before[3].flatten(), np.sum(before[3,0]), np.sum(before[3,1]), np.sum(before[3]))
    print("N After :", after[3].flatten(), np.sum(after[3,0]), np.sum(after[3,1]), np.sum(after[3]))
    print()
    print("Fz Before:", before[2].flatten(), np.sum(before[2]))
    print("Fz After :", after[2].flatten(), np.sum(after[2]))
    print()
    print("Fy Before:", before[1].flatten(), np.sum(before[1]))
    print("Fy After :", after[1].flatten(), np.sum(after[1]))
    print()
    print("Fx Before:", before[0].flatten(), np.sum(before[0]))
    print("Fx After :", after[0].flatten(), np.sum(after[0]))
    print()

#==========================#
# SUPPORT VECTOR REGRESSOR #
#==========================#
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
estimator = MultiOutputRegressor(SVR())
param_grid = {}

run_ML_model(estimator, param_grid, "Support Vector")

#=========================#
# DECISION TREE REGRESSOR #
#=========================#
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
estimator = BaggingRegressor(DecisionTreeRegressor())
param_grid={'n_estimators':[1,5,10]}

run_ML_model(estimator, param_grid, "Decision Tree")

#=========================#
# DECISION TREE REGRESSOR #
#=========================#
from sklearn.ensemble import RandomForestRegressor
estimator = RandomForestRegressor()
param_grid={'max_depth':[1,2,5]}

run_ML_model(estimator, param_grid, "Random Forest")

#===========================#
# ARTIFICIAL NEURAL NETWORK #
#===========================#
from sklearn.neural_network import MLPRegressor
estimator = MLPRegressor()
param_grid={
    'max_iter':[200],
    'solver':['adam', 'sgd'],
    'hidden_layer_sizes':[(16,16,16)],
}

run_ML_model(estimator, param_grid, "Artificial Neural Network")
