#   Loading the libraries
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 #import warnings
 #warnings.filterwarnings("ignore")
df = pd.read_csv("C:/Users/adm/Documents/Datasets/new dataset/heart.csv ")
df.head()
# Data exploration
df.shape
# Dataset has 303 columns and 14 columns
df.describe()
# The statistical details provides us with statistical information in numerical format. We can infer that in the AGE column the minimum age is 29 years and
# the max age is 77 years. The mean age is 54 years. Standard deviation and mean are statistical measures which give us an idea of the central tendency of the
# data. However, mean is affected by outliers hence we need more information to make accurate decision.
df.isnull().sum()
print(df.info())
# The dataset has no null values.

## Finding the correlation among attributes.
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot = True, cmap="terrain")

sns.pairplot(data = df)
df.hist(figsize=(12,12),layout=(5,3))

# Boxplot and whiskers plot
df.plot(kind = 'box',subplots = True, layout = (5,3),figsize = (12,12))
plt.show()

sns.catplot(data = df, x = "sex", hue ="output", palette ="husl")

sns.barplot(data = df , x = "sex" , y = 'chol', palette = 'husl')