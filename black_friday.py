import xgboost as xgb
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
import numpy as np
np.random.seed(123)
 
 
# --- lecture des sources 
train=pd.read_csv('xtrain.csv')
test=pd.read_csv('xtest.csv')

# --- separation des donnees features - labels
xtrain=train.drop({'Product_ID','Purchase'}, axis=1)
ytrain=train['Purchase']
xtest=test.drop({'Product_ID','Purchase'}, axis=1)
ytest=test['Purchase']

# --- presentation des donnees requise par xgboost
dtrain = xgb.DMatrix(xtrain,label=ytrain)  
dtest = xgb.DMatrix(xtest,label=ytest)  
#watchlist = [(dtest, 'eval'), (dtrain, 'train')]

# --- parametres du run
num_round = 400
param ={'tree_method': 'gpu_exact','objective' : 'reg:linear','eval_metric' :'rmse','max_depth' :10,'eta' : 0.1,'colsample_bytree' :0.5}
#  'gpu_exact' utilise CUDA  ---- mettre 'auto' pour n'utiliser que le CPU



# --- creation/calcul du modele
#xgb_model= xgb.train(param, dtrain, num_round,watchlist,early_stopping_rounds=10) avec watchlist
xgb_model= xgb.train(param, dtrain, num_round)  #sans watchlist


# ---  predictions
ypred = xgb_model.predict(dtest)


# --- performance
ypred=pd.DataFrame(ypred)
ypred.columns=['x']
ylabels=pd.DataFrame(ytest)
e=mean_squared_error(ylabels, ypred)
np.sqrt(e)
#2463.4022213202961


# --- validation croisee
xgbcv=xgb.cv(param, dtrain, num_boost_round=600,nfold=5,verbose_eval=1,seed=123,early_stopping_rounds=10)
# .....
# [330] train-rmse:2017.81+6.48927 test-rmse:2480.08+11.7391
num_round = 330
xgb_model = xgb.train(param, dtrain, num_round)
ypred = xgb_model.predict(dtest)

ypred = pd.DataFrame(ypred)
ypred.columns = ['x']
ylabels = pd.DataFrame(ytest)
e=mean_squared_error(ylabels, ypred)
np.sqrt(e)
#2462.3323366715476





