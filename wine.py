# Importing all the packages required

import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt 
import cufflinks as cf 

from sklearn import metrics
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

cf.go_offline()
%matplotlib inline


# call the data set
df = pd.read_csv('/Users/aneruthmohanasundaram/Desktop/New Project/wine.csv')

# To Print the head of the dataset (first ten values)
df.head(10)

# getting the info of the dataset 
df.info()

# describing the dataset 
df.describe()

###########################################################################################################################################
########################################################## Visualising the data ###########################################################
###########################################################################################################################################

# Basic Pairplot using seaborn 
plt.figure(figsize=(11,6))
sns.pairplot(df,diag_kind='hist')

# Quality vs fixed acidity
df.iplot(kind='bar',x='quality',y='fixed acidity')

# Quality vs volatile acidity
df.iplot(kind='bar',x='quality',y='volatile acidity')

# Quality vs citric acid
plt.figure(figsize=(11,8))
sns.barplot(x = 'quality', y = 'citric acid', data = df)

#  quality vs residual sugar
df.iplot(kind='bar',x='quality',y='residual sugar')

# quality vs sulphates
df.iplot(kind='histogram',x='quality',y='sulphates')

#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)

lq = LabelEncoder()

#Bad becomes 0 and good becomes 1 
df['quality'] = lq.fit_transform(df['quality'])

# counting the values 
df['quality'].value_counts()

# plotting the quality 
plt.figure(figsize=(10,7))
sns.countplot(df['quality'])

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

# createing a new DataFrame with respect to Coefficient
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