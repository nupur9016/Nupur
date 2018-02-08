#!/usr/bin/python


import sys
import pickle
import matplotlib
import random
import pandas as pd
import numpy as np
from numpy import mean
from time import time
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from tester import dump_classifier_and_data,test_classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
email_features=['from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages']

financial_features=[ 'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value']

target_label='poi'
features_list=[target_label]+financial_features+email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

## Exploring the data
#print(data_dict.keys())
print ('Total number of data points : %d'%len(data_dict.keys()))
print ('Number of Features: %d' %len(data_dict[data_dict.keys()[0]]))

number_of_poi=0
for person in data_dict.values():
    if person['poi']==1:
        number_of_poi +=1
print ('Number of POIs: %d' % number_of_poi)

###Dealing with missing/NaN values
# Transform data from dictionary to the Pandas DataFrame to 
df = pd.DataFrame.from_dict(data_dict, orient = 'index')
df = df.replace('NaN', np.nan)
#df.info()

print "Amount of NaN values in the dataset: ", df.isnull().sum().sum()

# Replacing 'NaN' in financial features with 0
df[financial_features] = df[financial_features].fillna(0)

imp = Imputer(missing_values='NaN', strategy='median', axis=0)

#impute missing values of email features 
df.loc[df[df.poi == 1].index,email_features] = imp.fit_transform(df[email_features][df.poi == 1])
df.loc[df[df.poi == 0].index,email_features] = imp.fit_transform(df[email_features][df.poi == 0])


### Task 2: Remove outliers
def VisualiseOutlier(data_dict,feature_x,feature_y):

    data=featureFormat(data_dict,[feature_x, feature_y,'poi'])
    for point in data:
        x=point[0]
        y=point[1]
        poi=point[2]
        if poi:
            color='red'
        else:
            color='blue'
        matplotlib.pyplot.scatter(x,y,color=color)
    matplotlib.pyplot.xlabel(feature_x)
    matplotlib.pyplot.ylabel(feature_y)
    matplotlib.pyplot.show()

#VisualiseOutlier(data_dict,"salary","bonus")

for key in data_dict:
    val=data_dict[key]['bonus']
    if val!='NaN':
        if val>=10000000:
            print key
##
###VisualiseOutlier(data_dict,'from_poi_to_this_person','from_this_person_to_poi')

for key in data_dict:
    val=data_dict[key]['from_poi_to_this_person']
    if val!='NaN':
        if val>=500:
            print key

outliers=['TOTAL','LAVORATO JOHN J','THE TRAVEL AGENCY IN THE PARK']
df = df.drop(outliers,0)


### Visualising after ouliers are removed

#VisualiseOutlier(data_dict,"salary","bonus")

### Task 3: Create new feature(s)
df['fraction_of_email_from_poi'] = df.from_poi_to_this_person/ df.to_messages
df['fraction_of_email_to_poi'] = df.from_this_person_to_poi / df.from_messages

#clean all 'inf' values which we got if the person's from_messages = 0
df = df.replace('inf', 0)
my_dataset = df.to_dict('index')

temp_my_features_list=features_list+["fraction_of_email_from_poi","fraction_of_email_to_poi"]

###Feature Selection using SelectKBest

num_features=15

def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    for key in k_best_features:
       print key,k_best_features[key]
   # print "{0} best features: {1}\n".format(k, k_best_features.keys())
    return k_best_features

best_features=get_k_best(my_dataset,temp_my_features_list,num_features)
my_feature_list=[target_label]+best_features.keys()


#my_feature_list=temp_my_features_list

## Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

###Scaling the features by MinMaxScaler
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)


#### Task 4: Try a varity of classifiers
##### We will iterate through variety of classifiers to see which one's prediction is the best.

##Splitting the data into train data and test data using cross validation

features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(features,labels, test_size=0.3, random_state=42)


### Create and test the KMeans Classifier
#clf = KMeans(n_clusters=2)


# Create and test the Decision Tree Classifier
clf = DecisionTreeClassifier(min_samples_split=14,criterion='entropy')
#clf = DecisionTreeClassifier()


### Create and test the Random Forest classifier
#clf = RandomForestClassifier()
#clf = RandomForestClassifier(max_features = 'log2',n_estimators = 30)

##clf.fit(features_train, labels_train)
##pred = clf.predict(features_test)
##accuracy = accuracy_score(pred,labels_test)
##print accuracy
##print 'precision = ', precision_score(labels_test,pred)
##print 'recall = ', recall_score(labels_test,pred)
##

##### Task 5: Tune your classifier to achieve better than .3 precision and recall 
##### using our testing script. Check the tester.py script in the final project
##### folder for details on the evaluation method, especially the test_classifier
##### function. Because of the small size of the dataset, the script uses
##### stratified shuffle split cross validation. For more info: 
##### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

##Parameter Tuning using GridSearchCV for Decision Tree

##grid = GridSearchCV(clf, param_grid={'min_samples_split':list(range(2,20,1)),'criterion':['gini','entropy']},cv=10, scoring='f1')
##grid.fit(features_train, labels_train)
##print(grid.best_score_)
##
##print(grid.best_params_)
##print(grid.best_estimator_)


##Parameter Tuning using GridSearchCV for Random Forest 

##grid = GridSearchCV(clf, param_grid={'n_estimators':[10,20,30],"max_features": ["auto", "sqrt", "log2"]},cv=10, scoring='f1')
##grid.fit(features_train, labels_train)
##print(grid.best_score_)
##
##print(grid.best_params_)
##print(grid.best_estimator_)

test_classifier(clf, my_dataset,my_feature_list, folds=1000)
##
##
#####Task 6
dump_classifier_and_data(clf, my_dataset, my_feature_list)
