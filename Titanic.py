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
plt.legend(['Percentage not  survived or unknown','Perentage of survived'])
plt.gcf().set_size_inches(12,6)
plt.show()