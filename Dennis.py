# Importing labriries
import numpy as np #LInear algebra
import pandas as pd #Data processing
import matplotlib.pyplot as plt #  for graphing
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold # from sklearn.cross_validation import KFold became outdated
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics

# Import data
data = pd.read_csv("C:/Users/adm/Documents/Datasets/cancer.csv",header=0)

print(data.head(2))

#Looking at the type of data that we have
data.info()
# Unnamed:32 have 0 non null object meaning it has null objects hence thos column cant be used
# Dropping column Unnamed:32
data.drop("Unnamed: 32",axis=1,inplace=True)
#Cheecking the column has been dropped
data.columns
# No Unnamed : 32 column

# Also id column isnt needed for our analysis
data.drop("id",axis=1,inplace=True)

# The data  can be divided into 3 parts
feature_mean=list(data.columns[1:11])
feature_se=list(data.columns[11:20])
feature_worst=list(data.columns[21:31])
print(feature_mean)
print(feature_se)
print(feature_worst)

# Starting with feature_mean
data["diagnosis"]=data["diagnosis"].map({'M':1,'B':0})

#Exploring the data
data.describe()
# getting the frequency of cancer  stages
sns.countplot(data['diagnosis'],label="Count")
# from the graph we can see that there is a number of begginig stage cancer that can be cured

# data analysis alittle feature selection
# drawing a correlation graph so as to remove multicolinierity  for feature mean
corr= data[feature_mean].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,cbar=True,square=True,annot=True,fmt='.2f',annot_kws={'size':15},
            xticklabels=feature_mean,yticklabels = feature_mean,
            cmap="coolwarm")

# the radius ,parameter and are  highly correlated as expected  from their  relation so from this we will use onyone of them
# compactness_mean ,concavity mean are highly correlated so we use compactness_mean here.
# the selected parameter for use is perimeter_mean ,texture_mean ,symmetry_mean

prediction_var =["texture_mean","perimeter_mean","smoothness_mean","compactness_mean","symmetry_mean"]

# splitting data into train and split data.
train, test =train_test_split(data,test_size=0.3)
#checking the dimension
print(train.shape)
print(test.shape)

train_x=train[prediction_var] # taking the training data input.
train_y=train.diagnosis # output of the training data

test_x=test[prediction_var]
test_y=test.diagnosis

model=RandomForestClassifier(n_estimators=100)
model.fit(train_x,train_y) # fitting the model for RF

# Predition
prediction =model.predict(test_x)
# prediction will contain the predicted value by our model  predicted values of disgnosis column for test inputs
metrics.accuracy_score(prediction,test_y)

# The accuracy here is 91%

#SVM
model1 = svm.SVC()
model.fit(train_x,train_y)
prediction1=model.predict(test_x)
metrics.accuracy_score(prediction1,test_y)
# svm has an accuracy of 93%.

# carrying out for all feature_mean
prediction_var=feature_mean

train_x=train[prediction_var]
train_y=train.diagnosis
test_x=test[prediction_var]
test_y=test.diagnosis

model = RandomForestClassifier(n_estimators=100)
model.fit(train_x,train_y)
prediction=model.predict(test_x)
metrics.accuracy_score(prediction,test_y)

# Checking for the importance of features in  prediction
feat_imp=pd.Series(model.feature_importances_,index=prediction_var).sort_values(ascending=False)
print(feat_imp)


# SVM Using all features
model3 = svm.SVC()
model3.fit(train_x,train_y)
prediction3=model3.predict(test_x)
metrics.accuracy_score(prediction3,test_x)
# The accuracy of  svm is much decreased.
# Checking for the importance


