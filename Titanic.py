import numpy as np
import pandas as pd
# importing the data
train=pd.read_csv("C:/Users/adm/Documents/Datasets/new dataset/train.csv")
test=pd.read_csv("C:/Users/adm/Documents/Datasets/new dataset/test.csv")
submission=pd.read_csv("C:/Users/adm/Documents/Datasets/new dataset/submission.csv")
train.columns
import seaborn as  sns
from matplotlib import pyplot as plt

# Look at missing values
train.isnull().sum()
sns.heatmap(train.isnull(),cbar=True).set_title("Missing Values heatmap")
plt.gcf().set_size_inches(16,6)
# Column Age has 177 null values,column cabin has687 null values  and column embarked has 2 null values
train.nunique()
train.describe()

# Features
# 1. survived
x =train["Survived"].value_counts().index
y =train["Survived"].value_counts()
plt.style.use("fivethirtyeight")
myexplode=(0.0,0.1)
mylabel=["Not survived(0)","Survived(1)"]
colors=["#f4acb7","#9d8189"]
plt.pie(y,labels=mylabel,autopct="%1.1f%%",startangle=15,shadow=True,explode=myexplode,colors=colors)
plt.gcf().set_size_inches(12,6)
plt.show()
# more than 60% of passengers died

# 2.Pclass
hue_color={0:'#012a4a',1:'#2c7da0'}
Pclass=['Class1','Class2','Class3']
plt.style.use('fivethirtyeight')
ax=sns.countplot(data=train,x="Pclass",hue="Survived",palette=hue_color)
plt.xticks(ticks=[0,1,2],labels=Pclass)
plt.legend(['Percentage not  survived or unknown','Percentage of survived'])
plt.gcf().set_size_inches(12,6)
plt.show()

# 3.sex
y=train["Sex"].value_counts()
myexplode=(0.0,0.1)
plt.style.use("fivethirtyeight")
mylabel=["Male","Female"]
colors=["#E63946","#F1FAEE"]
plt.pie(y,labels=mylabel,autopct="%1.1f%%",startangle=15,shadow=True,explode=myexplode,colors=colors)
# Approximately 65% of the tourist were male while the remaining 35% were  female

hue_color={0:"#8D99AE",1:'#ef233c'}
sex=['Male','Female']
plt.style.use("fivethirtyeight")
ax=sns.countplot(data=train,x="Sex",hue="Survived",palette=hue_color)
plt.xticks(ticks=[0,1],labels=sex)
plt.legend(['Percentage not  survived or unknown','Percentage of survived'])
plt.gcf().set_size_inches(12,6)
# More males died as  compared to females

# 4. Age
sns.countplot(x=train['Survived'],hue=pd.cut(train['Age'],5))
plt.style.use("fivethirtyeight")
plt.gcf().set_size_inches(12,6)
# A larger fraction of children under 16 survived than died
# The other age group the number of diedwas hier then the number of survivors


# 5. sibsp
y=train["SibSp"].value_counts()
myexplode=(0.0,0.1,0.2,0.4,0.1,0.3,0.4)
plt.style.use("fivethirtyeight")
mylabel=[0,1,2,3,4,5,8]
colors=["blue","steelblue","khaki","purple","yellow","darkgreen"]
plt.pie(y,labels=mylabel,autopct="%1.1f.%%" ,#used to label the wedges with their numeric value.
startangle=15,explode=myexplode,#array which specifies the fraction of the radius with which to offset each wedge.
colors=colors)
plt.axis("equal")
plt.gcf().set_size_inches(12,6)
plt.show()
# 91% of people travelled alone or with one of their sibling or spouse

hue_color={0:'#555b6e',1:'#890ae'}
plt.style.use("fivethirtyeight")
ax=sns.countplot(data=train,x='SibSp',hue='Survived')
plt.xticks(ticks=[0,1],labels=sex)
plt.legend(['Percentage not  survived or unknown','Percentage of survived'])
plt.gcf().set_size_inches(12,6)
# Survivalchances dropped drastically if someone travelled with more than two siblings

# Parch
hue_color={0:'#585123',1:'#eec170'}
plt.style.use("fivethirtyeight")
ax=sns.countplot(data=train,x='Parch',hue='Survived',palette=hue_color)
# plt.xticks(ticks = [0,1], labels = Sex)
plt.legend(['Percentage not survived or unknown', 'Percentage of survived'])
plt.gcf().set_size_inches(12,6)
plt.show()
#Similarily like 'sibsp', chances of survival dropped drastically if someone traveled with more than 2 parents/children along with one of the traveler

# we take the values of bins from the statistical analysis of colum 'Fare'
train['Fare_Category'] = pd.cut(train['Fare'], bins=[0,7.90,14.45,31.28,120], labels=['Low','Mid','High_Mid','High'])

hue_color={0:'#14213d',1:'#fca311'}
Fare_category=['Low','Mid','High_Mid','High']
plt.style.use("fivethirtyeight")
ax=sns.countplot(data=train,x='Fare_Category',hue='Survived',palette=hue_color)
plt.legend(['Percentage not survived or unknown', 'Percentage of survived'])
plt.gcf().set_size_inches(12,6)
plt.show()

# Embarked
Y=train["Embarked"].value_counts()
myexplode=(0.0,0.1,0.0)
plt.style.use("fivethirtyeight")
mylabel=['Southampton','Cherbourg','Queenstown']
colors = ['#a5a58d', '#6b705c','#3f4238']
plt.pie(Y,labels=mylabel,autopct="%1.1f%%",startangle=15,shadow=True,explode=myexplode,colors=colors)
plt.gcf().set_size_inches(12,6)
plt.show()
# 72% of the people boarded from Southampton. 20% boarded from Cherbourg and the rest boarded from Queenstown.

sns.countplot(x=train['Embarked'],hue=train['Survived'])
plt.style.use("fivethirtyeight")
plt.gcf().set_size_inches(12,6)
# People who boarded from Cherbourg had a higher chance of survival than people who boarded from Southampton or Queenstown.

# Data Imputation
train = train.iloc[:,:12]
train
train.drop("Cabin",axis=1,inplace=True)
# Creating a feature salutation to fill the null values of age column
import re
def split_it(data):
    result = re.search('^.*,(.*)\.\s.*$', data)
    if result.group(1) not in [' Mr', ' Miss', ' Mrs', ' Master']:
        return ' Misc'
    else:
        return result.group(1)

train['Salutation'] = train['Name'].apply(split_it)

test["Salutation"]=train['Name'].apply(split_it)
train

# Using salutation feature to fill null values according to title
train["Age"].fillna(train.groupby("Salutation")["Age"].transform('median'),inplace=True)

#Creating new feature surname to fill null values for embarked column
train["Surname"]=train.Name.map(lambda x:x.split(',')[0])
train

train[train.Embarked.isna()]
to_map = {'Icard': 'C',
          'Stone': 'S'}

train.Embarked.fillna(train.Surname.map(to_map),inplace = True)
train['Embarked'].replace({'S':1,'C':2,'Q':3},inplace=True)
train.isnull().sum()
train['Sex'] # Change the values of  sex column into categorical values
train["Sex"].replace({'male':0,'female':1},inplace=True)
train.drop(['Ticket','PassengerId','Name','Salutation',"Fare",'Surname'],axis=1,inplace=True)
train

test
test["Age"].fillna(test.groupby("Salutation")["Age"].transform("median"), inplace=True)
test['Embarked'].replace({'S':1,'C':2,'Q':3},inplace=True)
test.isnull().sum()
test['Sex'].replace({'male':0,'female':1},inplace=True)
test.drop(['Ticket','PassengerId','Name','Cabin','Salutation',"Fare"],axis=1,inplace=True)
test

#Model selection
X=train.drop(["Survived"],axis="columns")
y=train["Survived"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model1 = LogisticRegression()
model1.fit(X_train,y_train)
pred = model1.predict(X_test)
log_acc = accuracy_score(pred,y_test)
print(log_acc)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators=50)
model2.fit(X_train,y_train)
pred_2 = model2.predict(X_test)
rcf = accuracy_score(pred,y_test)
print(rcf)

#Decision Tree
from sklearn import tree
model3=tree.DecisionTreeClassifier()
model3.fit(X_train,y_train)
pred_3=model3.predict(X_test)
dtc=accuracy_score(pred_3,y_test)
print(dtc)

#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
model4=KNeighborsClassifier(n_neighbors=9)
model4.fit(X_train,y_train)
pred_4=model4.predict(X_test)
knn = accuracy_score(pred_4,y_test)
print(knn)


# Comparing accuracy of different models
print("Accuracy of Logistic Regression : ",log_acc)
print("Accuracy of Decision Tree Classifier : ",dtc)
print("Accuracy of KNN Classifier : ",knn)
print("Accuracy of Random Forest Classifier : ",rcf)
models_acc=[log_acc,dtc,knn,rcf]
names_of_models=['LogisticRegression','DecisionTreeClassifier','KNearestNeighbour','RandomForestClassifier']

sns.barplot(y=names_of_models,x=models_acc)
plt.gcf().set_size_inches(8,4)
plt.xlim(0.6,1.0)
plt.title("Model Accuracy")

# The best model was Logistic regresion
model1.fit(x,y)
pred=model1.predict(test)

submission=pd.DataFrame({"PassengerId": submission["passagerId"],"Survived":pred})
submission

submission.to_csv('submission.csv',index=False)
