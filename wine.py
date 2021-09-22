'''
The main of this project is to find the quality of the wine.
This is done using various classification algorithm.
'''

# Importing all the packages required

import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# call the data set
df = pd.read_csv('/Users/aneruthmohanasundaram/Desktop/New Project/wine.csv')

# To Print the head of the dataset (first ten values)
print(df.head(10))

# getting the info of the dataset 
print(f'\n{df.info()}')

# describing the dataset 
print(f'\n{df.describe()}')

###########################################################################################################################################
########################################################## Visualising the data ###########################################################
###########################################################################################################################################

# Basic Pairplot using seaborn 
plt.figure(figsize=(11,6))
sns.pairplot(df,diag_kind='hist')
plt.show()

# Quality vs fixed acidity
sns.barplot(data=df,x='quality',y='fixed acidity')
plt.show()

# Quality vs volatile acidity
sns.barplot(data=df,x='quality',y='volatile acidity')
plt.show()

# Quality vs citric acid
plt.figure(figsize=(11,8))
sns.barplot(x = 'quality', y = 'citric acid', data = df)
plt.show()

#  quality vs residual sugar
sns.barplot(data=df,x='quality',y='residual sugar')
plt.show()

# quality vs sulphates
sns.barplot(data=df,x='quality',y='sulphates')
plt.show()

# Making a binary classification system for the response variable. 
#Dividing wine into excellent and terrible categories by setting a quality limit.
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)

lq = LabelEncoder()

#Bad becomes 0 and good becomes 1 
df['quality'] = lq.fit_transform(df['quality'])

# counting the values 
# To check the qulity count range between two classification
print(df['quality'].value_counts())

# plotting the quality 
plt.figure(figsize=(10,7))
sns.countplot(df['quality'])
plt.show()

# To check correlation map
cor = df.corr()

# Plotting correlation map
plt.figure(figsize=(15,8))
plt.title('Confusion matrix of BCCD Dataset')
sns.heatmap(cor, annot = True)
plt.show()

###########################################################################################################################################
########################################################## Processing the models ##########################################################
###########################################################################################################################################

# Splitting the data into two sets 

X = df.drop('quality',axis=1)

y = df['quality']

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.28, random_state=101)

############################################################ Linear Regression ############################################################

# calling the linear regression classifier 
lr = LinearRegression()

# fit the data
lr.fit(X_train,y_train)

# creating a new DataFrame with respect to Coefficient
lr_coeff = pd.DataFrame(lr.coef_,X.columns,columns=['Coefficient'])
print('The coefficient of the data is:' + '\n\n',lr_coeff)

######################################################### Predicting the model ############################################################

# predicting the test data 
predict = lr.predict(X_test)

# To compare the predictions with the actual y_test value 
plt.figure(figsize=(10,5))
sns.scatterplot(y_test,predict)

# To analyse our residual distribution 
sns.histplot((y_test-predict),kde=True,bins=50) 

# Regression Evaluation Matrices 
print('Regression Evaluation Matrices for Linear Regression')
print('\n')
# To print the Mean Absolute Error Method 
print("Mena Absolute Error of our dataset is:",(metrics.mean_absolute_error(y_test,predict).round(2))*100)

print('\n')

# To print the Mean Square Error Method 
print("Mean Square Error of our dataset is:",(metrics.mean_squared_error(y_test,predict).round(2))*100)

print('\n')

# To print the Root Mean Square Error Method
print('Root Mean Square Error of our dataset is:',(np.sqrt(metrics.mean_squared_error(y_test,predict)).round(2))*100)

############################################################## Random Forest ############################################################

# Initialising the model 
rf = RandomForestClassifier(n_estimators=200)

# fitting the model 
rf.fit(X_train,y_train)

# To print the predcitions of the Random Forest Model 
predicti = rf.predict(X_test)
print(predicti)

# To print the classification report and confusion matrix for the Random Forest 
print('Report for Random Forest Model')
print('\n')
# To print the confusion matrix
print('The confusion matrix is: ' + '\n \n',confusion_matrix(y_test,predicti))

print('\n')

# To print the classification report 
print('The classification report is: ' + '\n \n',classification_report(y_test,predicti))

# Hyperparameter tuning out model --> this is done by selecting a specific parameter to fetch the optimised result
hyperList = [i for i in range(100,510,10)]
hyperOutput = []
for val in hyperList:
    rfHyper = RandomForestClassifier(n_estimators=200)
    rfHyper.fit(X_train,y_train)
    predictHyper = rf.predict(X_test)
    print(f'Accuracy score at {val}th estimator is {round(accuracy_score(y_test,predictHyper)*100,2)}')
    hyperOutput.append(round(accuracy_score(y_test,predictHyper)*100,2))

# Plotting the hyper parameter values
mapList = [i for i in range(100,510,10)]
plt.figure(figsize=(15,8))
plt.title('Hperparameter graph for Ranndom Forest')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy Score')
plt.plot(mapList,hyperOutput,marker='o')

# Cross Validation for Random Forest
rfc_eval = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = 10)
rfc_eval.mean()

######################################################## Support Vector Machine ########################################################

# Intialisng the SVM 

sv = SVC()

# Fitting the model 
sv.fit(X_train,y_train)

# To print the predcitions of the SVM Model 
predictio = sv.predict(X_test)
print(predictio)

# To print the classification report and confusion matrix for the Random Forest 

# To print the confusion matrix
print('The confusion matrix is: ' + '\n \n',confusion_matrix(y_test,predictio))

print('\n')

# To print the classification report 
print('The classification report is: ' + '\n \n',classification_report(y_test,predictio))

'''
Performing grid search which is a type of cross validation for Support Vector Machine
C is a hypermeter which is set before the training model and used to control error.
Gamma is also a hypermeter which is set before the training model and used to give curvature weight of the decision boundary.
More gamma more the curvature and less gamma less curvature.
'''
# Initialise the gird search variable 
pg = {"C":[0.1,1,10,100,1000],"gamma":[1,.1,.01,.001,.0001]}

# feed the search variable to the grid search 
grid = GridSearchCV(SVC(),pg,verbose=3,scoring='accuracy')

# Grid Search fixable
grid.fit(X_train,y_train)

# to get the best parameter fit from the grid
print('The best parameter is:',grid.best_params_)
print('\n')
# to get the best estimator from the grid
print('The best estimator is:',grid.best_estimator_)
print('\n')
# to get the best score from the grid
print('The best score is:',((grid.best_score_)*100).round(2),'%')
print('\n')
# to get the predictions from the grid
grid_predict = grid.predict(X_test)
print('The Predicted value is:',grid_predict)

# To print the classification report and confusion matrix for the SVM
print('Report for Supoort Vector Machine')
print('\n')
# To print the confusion matrix
print('The confusion matrix is:' + '\n \n',confusion_matrix(y_test,grid_predict))

print('\n')

# To print the classification report
print('The classification report is:' + '\n \n',classification_report(y_test,grid_predict))

# Result anlysis
plt.figure(figsize=(16,8))
plt.title('Graph to compare accuracy score for BCCD dataset')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy Score')
plt.bar(['Random Forest','Support Vector machine'],[round(accuracy_score(y_test,predicti)*100,2),round(accuracy_score(y_test,grid_predict)*100,2)])