import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize']=(20.0,10.0)
plt.rcParams['font.family']='serif'
df = pd.read_csv("C:/Users/adm/Documents/Datasets/new dataset/heart.csv ")
df.head()
df.columns
# convert from wide to long format and reMOVE all null values

dff = df.melt(id_vars=['output'],
              value_vars=['age','sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh','exng', 'oldpeak', 'slp', 'caa',
                          'thall', 'output'],
              value_name ='count')
dff.head()
# Basic  PLOT
p = sns.boxplot(data=dff,x= 'variable',y='count')
# Changing the order of the categories
sns.boxplot(data=dff,
            x='variable',
            y='count',
            order=sorted(dff.variable.unique()))

# Change orientation
sns.boxplot(data=dff,
            x='count',
            y='variable',
            order=sorted(dff.variable.unique()),
            orient = 'h')

# Desaturate
sns.boxplot(data=dff,
            x='variable',
            y='count',
            order=sorted(dff.variable.unique()),
            saturation=.25)
# Change the size of outlier marker
sns.boxplot(data=dff,
            x='variable',
            y='count',
            order=sorted(dff.variable.unique()),
            fliersize =20)
####
sns.set(rc= {"axes.facecolor":"#ccddff",
             "axes.grid":False,
             'axes.labelsize':30,
             'figure.figsize':(20.0,10.0),
             'xtick.labelsize':25,
             'ytick.labelsize':20})
h= sns.boxplot(data=dff,
               x='variable',
               y='count',
               palette='Paired',
               order=sorted(dff.variable.unique()),
               notch=True)
plt.xticks(rotation =45)
l = plt.xlabel('')
plt.ylabel("Results")
plt.text(5.4,200,"Box Plot", fontsize=70,color='black',fontstyle='italic')