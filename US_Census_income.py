##################################################################################################################
##################################################################################################################
################################# Create an algorithm to predict the US Adult Income ############################
##################################################################################################################
##################################################################################################################


# First import the necessary libraries and packages
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


# Upload the dataset
df = pd.read_csv("adult.csv")
df2 = pd.read_csv("adult.csv")


# Have a view on the dataset
df.head()


# Using df.head() we can see that there are some missing values in the dataset. The observation
# for each parameter instead of being empty is filled with a question mark. Thus, we should
# replace them.
df.replace('?', np.nan, inplace=True) 


# Clean the dataset from missing values
df.dropna(inplace = True)

    
# Our task is to predict the income of a person based on certain parameters. By viewing the
# dataset, we see that there are certain columns can be removed since they are not that 
# important.     
# Remove columns you don't need
df.drop('education', axis = 1, inplace = True)         
#df.drop('capital.gain', axis = 1, inplace = True)            
#df.drop('capital.loss', axis = 1, inplace = True)
#df.drop('fnlwgt', axis = 1, inplace = True)
#df.drop('workclass', axis = 1, inplace = True)
df.drop('relationship',axis = 1, inplace = True)
#df.drop('native.country',axis = 1, inplace = True)


# There are certain categorical variables in the dataset. We will use the get_dummies function
# for replacing the string values they have with integers (i.e., female = 1 male = 0).


df = pd.concat([df, pd.get_dummies(df['marital.status'],prefix = 'marital.status',prefix_sep = ':')], axis = 1)
df.drop('marital.status',axis = 1,inplace = True)

df = pd.concat([df, pd.get_dummies(df['occupation'],prefix = 'occupation',prefix_sep = ':')], axis = 1)
df.drop('occupation',axis = 1,inplace=True)

df = pd.concat([df, pd.get_dummies(df['race'],prefix = 'race',prefix_sep = ':')], axis = 1)
df.drop('race',axis = 1,inplace = True)

df = pd.concat([df, pd.get_dummies(df['sex'],prefix = 'sex',prefix_sep = ':')], axis = 1)
df.drop('sex',axis = 1,inplace = True)

#df = pd.concat([df, pd.get_dummies(df['native.country'],prefix = 'native.country',prefix_sep = ':')], axis = 1)
df.drop('native.country',axis = 1,inplace = True)

df = pd.concat([df, pd.get_dummies(df['workclass'],prefix = 'workclass',prefix_sep = ':')], axis = 1)
df.drop('workclass',axis = 1,inplace = True)

#df = pd.concat([df, pd.get_dummies(df['education'],prefix = 'education',prefix_sep = ':')], axis = 1)
#df.drop('education',axis = 1,inplace = True)

# Have another view on the dataset
df.head()



# Separate independed and depended variables
X = np.array(df.drop(['income'], 1))  # Independed variables
y = np.array(df['income'])            # Depended variables

            
# Scale the independed variables
X = preprocessing.scale(X)



# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Below we will use various algorithms and check which one can give the best solution


################################################################################################################
################################################################################################################
############################################# Decision tree ####################################################
################################################################################################################
################################################################################################################

from sklearn.tree import DecisionTreeClassifier, export_graphviz, export
#from sklearn import metrics
from sklearn.metrics import accuracy_score

DT = DecisionTreeClassifier(criterion = 'gini', max_depth = 5, min_samples_leaf = 1, min_samples_split  = 2)
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)


# Results

# Get the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))


accur_DT = accuracy_score(y_test, y_pred)
print("The Accuracy for Decision Tree Model is {}".format(accur_DT))


# Get the classification report for the decision tree algorithm
CL_report_DT = classification_report(y_test, y_pred)
print(classification_report(y_test, y_pred))


#################################################################################################################
#################################################################################################################
################################################## Random Forest ###############################################
##################################################################################################################
##################################################################################################################


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
RF = RandomForestClassifier(max_features = 'sqrt', max_depth = 29, min_samples_leaf = 3, min_samples_split = 11)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict(X_test)


# Results

# Get the confusion matrix for the decision matrix from the Random Forest algorithm
cm_RF = confusion_matrix(y_test, y_pred_RF)

# Get the classication report from the Random Forest algorithm
CL_report_RF = classification_report(y_test,y_pred_RF)
print(classification_report(y_test,y_pred_RF))

# Get the accuracy from the model
accur_RF = accuracy_score(y_test, y_pred_RF)
print("The Accuracy for the Random Forest algorithm is {}".format(accur_RF))


################################################################################################################
################################################################################################################
############################################## Support Vector Machine ##########################################
################################################################################################################
################################################################################################################


from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', C = 3.2, gamma = 0.008)
svc.fit(X_train, y_train)
y_pred_SVC = svc.predict(X_test)

# Results 

# Get the confusion matrix
cm_SVC = confusion_matrix(y_test, y_pred_SVC)


# Get the classification report
CL_report_SVC = classification_report(y_test, y_pred_SVC)
print(classification_report(y_test, y_pred_SVC))

# Get the accuracy of the SVM algorithm
accur_SVM = accuracy_score(y_test, y_pred_SVC)
print("The Accuracy for SVM is {}".format(accur_SVM))


# Below I present an example of how you can tune your code and calculate the optimal parameters

# Applying grid-search to find the best model and parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [3.1,3.2,3.4], 'kernel': ['rbf'], 'gamma' : [0.01,0.008,0.007,0.009]}]
grid_search = GridSearchCV(estimator = svc, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 3)

grid_search = grid_search.fit(X_train, y_train)

# Get the best accuracy and the best parameters
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_