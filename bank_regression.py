# Logistic regression implemented on UCI Bank Marketing dataset
# https://archive.ics.uci.edu/ml/datasets/bank+marketing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # Split data
from sklearn.preprocessing import LabelEncoder # Need to encode string data for regression 
from sklearn.metrics import confusion_matrix, classification_report

bank = pd.read_csv("banking.csv")

bank = bank.replace('unknown',np.NaN) # Replace all unknown values with NaN
bank = bank.dropna(axis=0) # Drop all rows with NaN

print("Dimensions:",bank.shape,'\n') # Shape prints dimensions
print("Features:",bank.columns.values)

bank.head()

# Simplify dataset

bank.loc[bank['education'] == 'basic.4y', 'education'] = 'basic' # Group basic educations together
bank.loc[bank['education'] == 'basic.6y', 'education'] = 'basic'
bank.loc[bank['education'] == 'basic.9y', 'education'] = 'basic'

bank.loc[bank['job'] == 'admin', 'job'] = 'white-collar' # Group white-collar jobs together
bank.loc[bank['job'] == 'management', 'job'] = 'white-collar'
bank.loc[bank['job'] == 'entrepreneur', 'job'] = 'white-collar'
bank.loc[bank['job'] == 'technician', 'job'] = 'white-collar'

bank.loc[bank['job'] == 'services', 'job'] = 'blue-collar' # Group blue-collar/service jobs together
bank.loc[bank['job'] == 'housemaid', 'job'] = 'blue-collar'
bank.loc[bank['job'] == 'services', 'job'] = 'blue-collar'

# Data exploration

bank.describe()

pd.DataFrame.hist(bank,column='y',bins=10) # Histogram of y values

count = 0
for i in bank['y']:
    if i == 1:
        count += 1
print("Proportion of subscriptions:",count/len(bank['y'])) # Proportion of 1 values
print("Proportion of no subscriptions:",1-count/len(bank['y'])) # Proportion of 0 values

months = {} # Plot subscriptions by month
for i, j in zip(bank['month'], bank['y']):
    if i not in months:
        months[i] = j
    else:
        months[i] += j
months = pd.Series(months)
months.plot.bar(grid=True)

jobs = {} # Plot subscriptions by job
for i, j in zip(bank['job'], bank['y']):
    if i not in jobs:
        jobs[i] = j
    else:
        jobs[i] += j
jobs = pd.Series(jobs)
jobs.plot.bar(grid=True)

ages = {} # Plot subscriptions by age
for i, j in zip(bank['age'], bank['y']):
    if i not in ages:
        ages[i] = j
    else:
        ages[i] += j
ages = pd.Series(ages)
ages.plot()

educations = {} # Plot subscriptions by education
for i, j in zip(bank['education'], bank['y']):
    if i not in educations:
        educations[i] = j
    else:
        educations[i] += j
educations = pd.Series(educations)
educations.plot.bar(grid=True)

# Logistic Regression Model Fitting
# Need to transform string data to numeric values before applying model

transformed = bank.select_dtypes(exclude=['number']) # Select all non-numeric data columns
transformed = transformed.apply(LabelEncoder().fit_transform) # Transform string values into numeric values
transformed = transformed.join(bank.select_dtypes(include=['number'])) # Join the newly encoded columns to the rest of the frame

x_train, x_test = train_test_split(transformed,test_size = 0.3) # Split data, 30% for testing
y_train, y_test = x_train['y'], x_test['y'] # Isolate the y values for training and testing
x_train, x_test = x_train.drop(['y'],axis=1), x_test.drop(['y'],axis=1) # Drop the y values from the input training data

model = LogisticRegression().fit(x_train,y_train) # Fit a model to the training data

predicted = model.predict(x_test) # Predict the values

count = 0
for i, j in zip(predicted,y_test): # Compare predicted values to actual values and count the errors
    if i == j:
        count += 1
print("Accuracy:",(count/len(y_test))*100,"%") # Print success rate

# Confusion Matrix & Classification Report
print(confusion_matrix (y_test,predicted))
print(classification_report(y_test,predicted,target_names=['0','1']))

