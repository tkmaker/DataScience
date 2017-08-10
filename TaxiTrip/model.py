# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 07:34:40 2017

@author: takalyan
"""

# Import libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split



# Load training data as train
train_df = pd.read_csv('train.csv')

# Load testing data as test
test_df = pd.read_csv('test.csv')

#Remove outliers
train_df = train_df[train_df.trip_duration < 500000]



# First combine the data for easy processing
df_all = pd.concat([train_df, test_df], axis=0)

df_all['pickup_datetime'] = df_all['pickup_datetime'].apply(pd.Timestamp)
df_all['dropoff_datetime'] = df_all['dropoff_datetime'].apply(pd.Timestamp)

df_all['trip_duration_log'] = df_all['trip_duration'].apply(np.log)

# Feature Extraction
X = np.vstack((df_all[['pickup_latitude', 'pickup_longitude']], 
               df_all[['dropoff_latitude', 'dropoff_longitude']]))

# Remove abnormal locations
min_lat, min_lng = X.mean(axis=0) - X.std(axis=0)
max_lat, max_lng = X.mean(axis=0) + X.std(axis=0)
X = X[(X[:,0] > min_lat) & (X[:,0] < max_lat) & (X[:,1] > min_lng) & (X[:,1] < max_lng)]

pca = PCA().fit(X)

df_all['pickup_pca_lat'] = pca.transform(df_all[['pickup_latitude', 'pickup_longitude']])[:,0]
df_all['pickup_pca_long'] = pca.transform(df_all[['pickup_latitude', 'pickup_longitude']])[:,1]

df_all['dropoff_pca_lat'] = pca.transform(df_all[['dropoff_latitude', 'dropoff_longitude']])[:,0]
df_all['dropoff_pca_long'] = pca.transform(df_all[['dropoff_latitude', 'dropoff_longitude']])[:,1]

# Distances

def arrays_haversine(lats1, lngs1, lats2, lngs2, R=6371):
    lats_delta_rads = np.radians(lats2 - lats1)
    lngs_delta_rads = np.radians(lngs2 - lngs1)
    
    a = np.sin(lats_delta_rads / 2)**2 + np.cos(lats1) * np.cos(lats2) * np.sin(lngs_delta_rads / 2)**2
    c = 2 * np.arcsin(a**0.5)
    
    return R * c

df_all['haversine'] = arrays_haversine(
    df_all['pickup_latitude'], df_all['pickup_longitude'], 
    df_all['dropoff_latitude'], df_all['dropoff_longitude'])



# Date-Time

df_all['month'] = df_all['pickup_datetime'].dt.month
df_all['weekofyear'] = df_all['pickup_datetime'].dt.weekofyear
df_all['weekday'] = df_all['pickup_datetime'].dt.weekday
df_all['hour'] = df_all['pickup_datetime'].dt.hour



# Cluster trips and aggregate information about them
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=32).fit(X)

df_all['pickup_cluster'] = kmeans.predict(df_all[['pickup_latitude', 'pickup_longitude']])
df_all['dropoff_cluster'] = kmeans.predict(df_all[['dropoff_latitude', 'dropoff_longitude']])


features = [
    'vendor_id', 'passenger_count',
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
    'pickup_pca_lat', 'pickup_pca_long', 'dropoff_pca_lat', 'dropoff_pca_long',
    'haversine',  'month', 'weekofyear', 'weekday', 'hour', 
   ]

# Train Models
xtrain = df_all[df_all['trip_duration'].notnull()][features].values
ytrain = df_all[df_all['trip_duration'].notnull()]['trip_duration_log'].values




# Split the training data to train and validate
X_train, X_valid, y_train, y_valid = train_test_split(xtrain, ytrain,
                                                    train_size=0.8, test_size=0.2)


xgb_params = {
    #'eta': 0.06, #learning rate.  shrinks the feature weights to make the boosting process more conservative.
    #'max_depth': 2, #maximum depth of a tree, increase this value will make the model more complex / likely to be overfitting. 
    #'min_child_weight': 4,
    'subsample': 0.5, #subsample ratio of columns for each split, in each level. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.
    'colsample_by_tree': 0.5,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}




# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_valid, y_valid)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=500, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=10, 
                   show_stdv=False
                  )



num_boost_rounds = len(cv_result)

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)



y_pred_val = model.predict(dval)
from sklearn.metrics import r2_score
r2 = r2_score(y_valid,y_pred_val)
print ("R2 Score=",r2)

#Get test set
X_test = df_all[df_all['trip_duration'].isnull()][features].values

dtest = xgb.DMatrix(X_test)
y_test = np.exp(model.predict(dtest))

dataset_test = pd.read_csv('test.csv')

# make predictions and save results
output = pd.DataFrame({'Id': dataset_test['id'], 'trip_duration': y_test})

#Write results to file
output.to_csv('tk_pred.csv', index=False)