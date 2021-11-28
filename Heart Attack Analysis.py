import numpy as np
import pandas as pd
df = pd.read_csv("C:/Users/adm/Documents/Datasets/new dataset/heart.csv ")

# data description
df.columns
print("Total rows and columns in dataset",df.shape)
# checking for the presence of null values
df.isnull().sum()
df.info()
# Checking for duplicates
df.duplicated().sum()

# There isone duplicate Drop the duplicate
df.drop_duplicates(inplace=True)
# 1  row deleted from dataset
df.shape

# Statistical analysis of data
df.describe()
df.caa.value_counts()
# Column caa has 4 entries with the value 4 whkich is outside the range ,so replace it with the mode of features
index=df[df["caa"]==4]
index

# the  4  values are added to values 0
from statistics import mode
mode = mode(df.caa)
df.loc[df.caa==4,'caa']=mode
df.caa.value_counts()

df.thall.value_counts()
# thall  has 2  entreis that are  0 which is out of range.Replace them with mode
Index=df[df['thall']==0]
from statistics import mode
Mode= mode(df.thall)
df.loc[df.thall == 0,'thall']=Mode
df.thall.value_counts()

# Data visualizaion
from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns

X = df["age"]
plt.hist(X,color='green')
plt.title('Age Distribution',fontsize=25)
plt.xlabel("Age",fontsize=20)
plt.ylabel("Frequency",fontsize=20)
plt.grid(True)
plt.gcf().set_size_inches(12,6)
plt.show()

# alterantively
sns.histplot(data=df,x='age',stat='probability',discrete=True,kde=True)

x=df["sex"].value_counts().index
print(x)
y=df['sex'].value_counts()
sns.countplot(data=df,x='sex')
plt.bar(x,y,color=["pink","black"])
plt.tight_layout()
plt.xlabel('Sex_category',color='black',fontsize=20)
plt.ylabel('Count',color='black',fontsize=20)
plt.title('Sex- distribution',color='red',size=25)
plt.gcf().set_size_inches(8,6)

# Types of chest pain (cp)
x=df['cp'].value_counts().index
x
y=df['cp'].value_counts()
y
myexplode=(0,0.1,0,0.1)
mylabel=  ['Typical angina(0)','atypival(1','Non_aniginal pain(2)','Asymptomatic pain(3)']
fig, ax=plt.subplots(figsize=(10,7))
plt.pie(y,labels=mylabel,autopct='%1.1f%%',startangle=15,shadow=True,explode=myexplode)
ax.set_title(" Pie chart of  distribution of Chest pains")
ax. legend(loc='upper right')
sns.countplot(data=df,x='cp')
plt.xlabel('Chest Pain')
plt.ylabel('Count')
plt.tight_layout()
plt.grid(True)
plt.gcf().set_size_inches(12,6)

# Correlation of features of  dataset with target variable
# 1. Age
plt.style.use('fivethirtyeight')
plt.title('Effect of age on blood pressure',fontsize=20)
sns.lineplot(df['age'],df['trtbps'])
plt.gcf().set_size_inches(12,6)
# with increase with age blood pressure increases.

plt.style.use('ggplot')
plt.title('Effect of age on cholestrol level',fontsize=20)
sns.lineplot(df['age'],df['chol'])
plt.gcf().set_size_inches(12,6)
# with increase in age, cholestrollevel increased.

# 5. cholestrol
ax = sns.kdeplot(data = df, x = 'chol', bw_adjust = 0.9, hue = 'output', palette = ['blue', 'yellow'])
ax.set(xlabel = 'cholestrol')
plt.legend(['high chance', 'low chance'])
plt.gcf().set_size_inches(12,6)
plt.show()


# models
# logistic Regresion
# features to train the model
X= df.iloc[:,:13]

# Target variable
Y=df.iloc[:,13:]

# importing libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

#logistic Regression
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
Y_train=np.ravel(Y_train)
clf=LogisticRegression()
clf.fit(X_train,Y_train)
pred=clf.predict(X_test)
log_acc=accuracy_score(pred,Y_test)
print("REPORT: ")
print(confusion_matrix(Y_test,pred))
print(classification_report(Y_test,pred))
plot_confusion_matrix(clf,X_test,Y_test)
plt.grid(False)
plt.gcf().set_size_inches(12,6)


# 2. Knn classifier
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=9)
clf.fit(X_train,Y_train)
pred=clf.predict(X_test)
knn=accuracy_score(pred,Y_test);
print("REPORT: ")
print(confusion_matrix(Y_test,pred))
print(classification_report(Y_test,pred))
plot_confusion_matrix(clf,X_test,Y_test)
plt.grid(False)
plt.gcf().set_size_inches(12,6)