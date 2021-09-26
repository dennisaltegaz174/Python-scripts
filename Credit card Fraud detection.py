# Import libraries
 import pandas as pd
 import  numpy as np
 import  matplotlib.pyplot as plt
 from numpy.random import RandomState
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import plot_confusion_matrix

# CLASSIFIERS FOR TRAINING
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.svm import SVC
 from sklearn.tree import DecisionTreeClassifier
 from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
 from sklearn.naive_bayes import GaussianNB
 from sklearn.discriminant_analysis
 import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
 from sklearn.linear_model import LogisticRegression

 # Data reading
 df = pd.read_csv("C:/Users/adm/Documents/Datasets/creditcard.csv")
 df.head(7)
 df.shape

 # convert from float64 to float32 to reduce memory size
 df =df.astype({col:'float32' for col in df.select_dtypes('float64').columns})
 df.dtypes

 #Visualizing the data
 count_classes=df['Class'].value_counts()
 plt.title("Records in each class")
 count_classes.plot(kind='bar',color='c')
 plt.xticks(rotation="horizontal")
 plt.xlabel("Class")
 plt.ylabel("count")
 plt.show()

 # checking how many records are in each class during the time
 # All time was divided into 4 different intervals
 bins=[0,max(df['Time'])/4,max(df['Time'])/2,3*max(df['Time'])/4,max(df['Time'])]
 time_interval=pd.cut(df['Time'],bins=bins)

 #Then we add group data by class
 df_grouped2 =df.groupby(['Class',time_interval]).size().reset_index(name='Count')
 df_grouped2
# barcharts of each class
 mask1=df_grouped2['Class']==0
 mask2=df_grouped2['Class']==1

 #applying masks
 df_sliced1 =df_grouped2.loc[mask1]
 df_sliced2=df_grouped2.loc[mask2]
 # Creating plot with 2 subplots
 fig, axes=plt.subplots(nrows=1,ncols=2,figsize=(10,3))

 # setting colormaps for each subplot
 color1=plt.cm.spring(np.linspace(0,1,len(df_sliced1['Time'].unique())))
 color2=plt.cm.winter(np.linspace(0,1,len(df_sliced1['Time'].unique())))

 #Drawing plots
 df_sliced1.plot(x="Time",y="Count",kind="bar",color=color1,title="Class 0",ax=axes[0])
df_sliced2.plot(x="Time", y="Count", kind="bar", color=color2, title="Class 1", ax=axes[1])

# Drawing boxplots to see how data is distributed
fig, axes=plt.subplots(nrows=1,ncols=2,figsize=(10,3))
# Title for plots
axes[0].title.set_text("Class 0")
axes[1].title.set_text("Class 1")
# color for drawing plots
color1=dict(boxes="pink")
color2=dict(boxes="tan")

bplot1=df_sliced1.boxplot(column=['Count'],grid=False,ax=axes[0],color=color1,patch_artist=True)
bplot1=df_sliced2.boxplot(column=['Count'],grid=False,ax=axes[1],color=color2,patch_artist=True)
plt.show()

#Normaliation on data
# applying min-max normalization
scaler=MinMaxScaler(feature_range=(0,1))
normed=scaler.fit_transform(df)
df_normed=pd.DataFrame(data=normed,columns=df.columns)
df_normed.head()

#splitting to train and val subsets
rng=RandomState()
train=df_normed.sample(frac=0.7,random_state=rng)
val=df_normed.loc[~df_normed.index.isin(train.index)]
train.reset_index(drop=True,inplace=True)
val.reset_index(drop=True,inplace=True)
train.head(3)
val.head(3)

# splitting dataset into input and output variables(x and y)
# We should predict 'class'
# forming x and y  data
x_column=df.columns[:-1]
y_column=df.columns[-1]
print(x_column)
# Creating x and y data for training
x_raw_train=train[x_column]
y_raw_train=train[y_column]

x_train=x_raw_train.copy()
y_train=y_raw_train.copy()

# Creating x and y data for validation
x_raw_val=val[x_column]
y_raw_val=val[y_column]

x_val =x_raw_val.copy()
y_val=y_raw_val.copy()

#Classification
# List of allclassifiers that will be used on the data
all_classifiers=[KNeighborsClassifier(2),
                 SVC(),
                 DecisionTreeClassifier(),
                 RandomForestClassifier(),
                 AdaBoostClassifier(),
                 GradientBoostingClassifier(),
                 GaussianNB(),
                 LogisticRegression()]

# storing accuracies to build plots  nad for choosing the best classifier
all_acc={}

#Learn all classifiers and save trained models in pickle_format
for classifier in all_classifiers:
 #get the classifier name
 classifier_name=classifier._class_._name_
      #train model
      model=classifier
      model.fit(X_train,Y_train)

     #validate model
     model_pred=model.predict(x_val)
     model_acc=accuracy_score(Y_val,model_pred)
     # calculate confusion matrix for train and val subsets
     fig,axes=plt.subplots(nrows=1,ncols=2,figsize(15,6))
     plt.suptitle.set_text(classifier_name,fontsize=14)
     axes[0].title.set_text("Confusion Matrix (Train)")
     axes[1].title.set_text("Confusion Matrix (Val)")
     plot_confusion_matrix(model,X_train,Y_train,cmap=plt.cm.RdPu,ax=axes[0])
     plot_confusion_matrix(model,X_val,Y_val,cmap=plt.cm.GnBu,ax=axes[1])

     #save model
     filename=classifier_name+'_model.pickle'
     pickle.dump(model,open(filename'wb'))
#load model
      load_model=pickle.load(open(filename,'rb'))
     result=loaded_model.score(x_val,y_val)

     #print results
     print()

