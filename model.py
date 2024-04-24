#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
df = pd.read_csv("data/train_data_titanic.csv")

#Observe data
df.head()
df.info()
df.groupby('Survived').mean()
df.isnull().sum()
df.isnull().sum()>(len(df)/2)
df['Sex'].value_counts()
df['Age'].isnull().value_counts()
df.groupby('Sex')['Age'].median().plot(kind='bar')
df['Embarked'].value_counts()
df.corr()
sns.pairplot(df[['Survived','Fare']], dropna=True)

#Remove not using columns
df.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)

#Fill the missing values
df['Age'] = df.groupby('Sex', group_keys=False)['Age'].apply(lambda x: x.fillna(x.median()))

df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(),inplace=True)

#Convert data type
df = pd.get_dummies(data=df,columns=['Sex','Embarked'])

#Reduce similar columns
df.drop('Sex_female', axis=1, inplace=True)

#Prepare training data
X = df.drop(['Survived'],axis=1)
y = df['Survived']

#Split data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=69)

#Using Logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)

#Evaluate
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score
accuracy_score(y_test,predictions)
recall_score(y_test,predictions)
precision_score(y_test,predictions)
pd.DataFrame(confusion_matrix(y_test,predictions),columns = ['Predictnot Survived', 'PredictSurvived'],index=['Truenot Survived','TrueSurvived'])

#Export model
import joblib
joblib.dump(lr, 'Titanic-LR-test1.pkl', compress=3)