import numpy as np
import pandas as pd
# importing the data
train=pd.read_csv("C:/Users/adm/Documents/Datasets/new dataset/train.csv")
test=pd.read_csv("C:/Users/adm/Documents/Datasets/new dataset/test.csv")
submission=pd.read_csv("C:/Users/adm/Documents/Datasets/new dataset/submission.csv")
train
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
