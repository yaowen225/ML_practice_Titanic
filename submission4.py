#import modules
import pandas as pd

#import dataset
df_train = pd.read_csv("titanic/train.csv")
df_test = pd.read_csv("titanic/test.csv")

#Observe data
df_train.head()
df_train.info()
df_train.isnull().sum()
df_test.isnull().sum()
df_train.isnull().sum()>(len(df_train)/2)
df_train['Sex'].value_counts()
df_train['Age'].isnull().value_counts()
df_train.groupby('Sex')['Age'].median().plot(kind='bar')
df_train['Embarked'].value_counts()
df_train.corr()

#Train part
#Remove not using columns
df_train.drop(['Name','Ticket','Cabin','Pclass','PassengerId'], axis=1, inplace=True)

#Fill the missing values
df_train['Embarked'].fillna(df_train['Embarked'].value_counts().idxmax(), inplace=True)
# df_train['Age'] = df_train.groupby('Sex', group_keys=False)['Age'].apply(lambda x: x.fillna(x.median()))
df_train['Age'].fillna(df_train['Age'].value_counts().idxmax(), inplace=True)

#Convert data type
df_train = pd.get_dummies(data=df_train, columns=['Sex','Embarked'])

#Reduce similar columns
df_train.drop('Sex_female', axis=1, inplace=True)

#Prepare training data
X = df_train.drop(['Survived'], axis=1)
y = df_train['Survived']

#Using Logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X,y)

#Test part
#Remove not using columns
df_test.drop(['Name','Ticket','Cabin','Pclass','PassengerId'], axis=1, inplace=True)

#Fill the missing values
df_test['Fare'].fillna(df_test['Fare'].value_counts().idxmax(), inplace=True)
# df_test['Age'] = df_test.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.median()))
df_test['Age'].fillna(df_test['Age'].value_counts().idxmax(), inplace=True)

#Convert data type
df_test = pd.get_dummies(data=df_test, columns=['Sex','Embarked'])

#Reduce similar columns
df_test.drop('Sex_female', axis=1, inplace=True)

#Predict
predictions = lr.predict(df_test)

#Prepare submission
SubmissionDF = pd.DataFrame(columns=['PassengerId','Survived'])
SubmissionDF['PassengerId'] = range(892,1310)
SubmissionDF['Survived'] = predictions

#Output
SubmissionDF.to_csv('submission4.csv', index=False)