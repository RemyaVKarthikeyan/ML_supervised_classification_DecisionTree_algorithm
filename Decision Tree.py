#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all required libraries
import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt

#Loading the dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
dataset

# retrieving the dimensions of the dataset
dataset.shape

dataset.head()
dataset.tail()

#printing a concise summary of the DataFrame
dataset.info()

#returning a statistical description of the data in the DataFrame
dataset.describe()

#returning a statistical description of the data in all columns of the DataFrame
dataset.describe(include='all')

#plotting the distribution of column Age
sns.histplot(dataset.Age)
plt.title('Age Distribution')
plt.show()

#explore the age distribution of the people 
#who have responded to the social media ads and bought the product and those who haven’t.
plt.figure(figsize=(10,5))
plt.title("Product purchasing based on the age")
sns.histplot(x='Age',hue='Purchased',data=dataset)
plt.show()

#converting 'Gender'to numerical value
dataset.Gender=dataset.Gender.replace(['Male','Female'],[0,1])
dataset.head()

#Categorical values can be converted to numerical values using the code below as well. 
#In this case, Ordinal encoder method converts Female to 0 value and Male to 1 value
#Code is given below:

#from sklearn import preprocessing
#enc=preprocessing.OrdinalEncoder()
#new_gender=enc.fit_transform(dataset[['Gender']])
#dataset.Gender=new_gender
#dataset.head()

# Classification using Decision algorithm

# (a) Determining the class feature and input features:

#In this dataset, 'Purchased' column is the class label or dependent variable,
#other columns (except UserId) are independent variables or input features.
#first step, slicing data into input and output
x=dataset.iloc[:,[1,2,3]].values
y=dataset.iloc[:,4].values

# (b) Splitting the dataset into the Training set and the Test set:

#data splitting into training data and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

# (c) Scaling features:

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_s=sc.fit_transform(x_train)
x_test_s=sc.transform(x_test)

# (d) Training the model:

#Fitting the decision tree classification to the training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train_s,y_train)

# (e) Evaluating the model:

#predicting the test set results
y_pred=classifier.predict(x_test_s)
print(y_pred)

print(y_test)

#evaluating the performance of the model
from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred)
print("accuracy:%.2f\n\n:"%(acc))
cm=metrics.confusion_matrix(y_test,y_pred)
print('Confusion Matrix:')
print(cm,'\n\n')
print('-------------------------------------')
result=metrics.classification_report(y_test,y_pred)
print('Classification Report:\n')
print(result)

#visualising the confusion matrix using seaborn heatmap
ax=sns.heatmap(cm,cmap='flare',annot=True,fmt='d')
plt.title('Confusion Matrix',fontsize=12)
plt.xlabel('Predicted Class',fontsize=12)
plt.ylabel('True Class',fontsize=12)
plt.show()



# In[ ]:




