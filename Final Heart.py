# loading basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Reading the csv file
df = pd.read_csv("C:/Users/adm/Documents/Datasets/new dataset/heart.csv ")
df.head()

# Exploring the  dataset so as derive useful info
df.columns
df = df.rename(columns = {'output':'target'}, inplace=False)
df.describe
# Statistical Details provide statistical information in numerical format. From age column we can infer that the minimum
# age is 29  years and the maximum is  is 77 years ,mean age is 54 years


df.isnull().sum()
print(df.info())
# The data has no null values

# Finding the correlation among attributes
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True,cmap='terrain')
sns.pairplot(data=df)
df.hist(figsize=(12,12),layout=(5,3));

# Boxplot and whiskers plot
df.plot(kind='box',subplots = True,layout = (5,3),figsize=(12,12))

sns.catplot(data=df,x='sex',y='age',hue='target',palette=['blue', 'yellow'])
sns.barplot(data=df,x='sex',y='chol',hue='target',palette=['darkred','steelblue'])
df['sex'].value_counts() # 207 males , 96 females
df['target'].value_counts() # 165 cases of heart disease
df['thall'].value_counts()
# Results of thallium stress test measuring blood flow to heart , with possible values of normal,fixed_defect,reversible_defect.
sns.countplot(x="sex",data=df,palette='husl',hue = 'target')
# with 1 here representing males and 0 females we observe that  females having heart disease are comparatively less when
# compared to males. Males have low heart disease compared to females in the dataset
sns.countplot(x='target',palette = 'BuGn',data=df)
# The count of not having having heart disease and  not having heart disease are almost balanced.Not having has a frequency
# of 140 and having is 160
plt.figure(figsize=(20,10))
sns.countplot(x='caa',hue='target',data=df)
# ca: number of major vessels(0-3) colored by  flourosopy
df['caa'].value_counts()
# caa has a negative correlation with the target i.e an increase in caa will lead to a drop in heart disease and vice versa
plt.figure()
plt.style.use('ggplot')
sns.countplot(x='thall',data= df,hue='target',palette = 'BuPu')
plt.title("Thall vs Target",fontsize = 15)
plt.gcf().set_size_inches(12,6)


df['cp'].value_counts()
plt.figure()
sns.countplot(x='cp',data=df,hue='target',palette='rocket')
plt.title("Chestpain vs Target")

plt.figure(facecolor="steelblue")
sns.boxplot(x="sex",y='chol',hue='target',palette='seismic',data=df)
plt.title('sex vs Target() \n Example \n',fontsize = 12,  fontweight='bold')
