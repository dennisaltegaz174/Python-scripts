import matplotlib.pyplot as plt
import pandas as pd
df=pd.DataFrame({'Name':{0:'John',1:'Bob',2:'sheila'},
                 'Courses':{0:'Master',1:'Graduate',2:'Graduate'},
                 'Age':{0:27,1:23,2:21}})
df
pd.melt(df)
pd.melt(df,id_vars=["Name","Courses"])

# Facet Grid
import seaborn as sns
tips=sns.load_dataset("tips")
tips
g=sns.FacetGrid(tips,col="time",row="sex")
g.map(sns.scatterplot,'total_bill','tip')

y=sns.FacetGrid(tips,col="time",row="sex",hue="sex")
y.map_dataframe(sns.scatterplot,x='total_bill',y='tip')

gg=sns.FacetGrid(tips,col='time',height=5)
gg.map(plt.hist,'total_bill')
tips.head()
p=sns.FacetGrid(tips,row='sex',col='time')


dt = pd.read_csv("C:/Users/adm/Documents/Datasets/heart_statlog_cleveland_hungary_final.csv")
dtt=pd.melt(dt)
dtt.head()
x=sns.FacetGrid(dtt,col='variable',height=4,col_wrap=4,ylim=(0,10),palette='pal')
x.map(plt.hist,'value')
p=sns.FacetGrid(dtt,col='variable',height=4,col_wrap=4,ylim=(0,10))
x.map(plt.boxplot,"value")

liver_data=pd.read_csv("C:/Users/adm/Documents/Datasets/Indian Liver Patient Dataset (ILPD).csv")
liver2=pd.melt(liver_data)
f=sns.FacetGrid(liver2,col="variable",height=4,col_wrap=4,ylim=(0,10))
f.map(plt.hist,'value')


# pivoting and unpivoting dataframes.
s = 'Tesla Model S P100D'
x = 'Tesla Model X P100D'
three = 'Tesla Model 3 AWD Dual Motor'

s_data = [s, 2.5, 2.51, 2.54]
x_data = [x, 2.92, 2.91, 2.93]
three_data = [three, 3.33, 3.31, 3.35]
data = [s_data,x_data, three_data]
df = pd.DataFrame(data , columns= ['car_model', 'Sept 1 9am', 'Sept 1 10am', 'Sept 1 11am'])
df
df_unpivoted  = df.melt(id_vars=['car_model'],var_name  ='date' , value_name= '0-60mph_in_seconds')
df_unpivoted
df_unpivoted.groupby('car_model')['0-60mph_in_seconds'].min()


# lambda functions
# Normal python function
def a_name(x):
    return x+x

# Lambda function
lambda x: x+x
# Scalar values
(lambda x: x*2)(12)
# List
list_1 = [1,2,3,4,5,6,7,8,9]
filter(lambda x:x%2 == 0,list_1)
list(filter(lambda x:x%2==0, list_1)) # returns list of even numbers

# map
list_1 = [1,2,3,4,5,6,7,8,9]
cubed = map(lambda x :pow(x,3), list_1)
list(cubed)

# Series  object
# A Series object is a column in a data frame, or put another way, a sequence of values with corresponding indices. Lambda functions can be used to manipulate values inside a Pandas dataframe
df = pd.DataFrame({
    'Name': ['Luke','Gina','Sam','Emma'],
    'Status': ['Father', 'Mother', 'Son', 'Daughter'],
    'Birthyear': [1976, 1984, 2013, 2016],
})
df
# Lambda with Apply() function by Pandas. This function applies an operation to every element of the column
df['age'] = df['Birthyear'].apply(lambda x : 2021 - x)
df
# Lambda with Python’s Filter()
list(filter(lambda x : x>18 , df['age']))

# Lambda with Map() function
# Double the age of everyone
df['double_age']=df['age'].map(lambda x:x*2)
df
# Conditional Lambda statement
df['Gender'] = df['Status'].map(lambda x:'Male', if x== 'father'or 'son' else 'female')
df
df['Gender'] = df['Status'].map(lambda x: 'Male' if x=='father' or x=='son' else 'Female')

# Filtering
number = [-2,-1,0,1,2]
# using a lambda function
positive_numbers = filter(lambda n:n>0,number)
positive_numbers
list(positive_numbers)

# Using a user-defined function
def is_postive(n):
    return  n > 0
list(filter(is_postive, number))

def identity(x):
    return x

identity(42)
objects = [0, 1, [], 4, 5, "", None, 8]
list(filter(identity,objects))

# Filtering Iterables With filter()
# Extracting Even Numbers
numbers = [1, 3, 10, 45, 6, 50]

def extract_even(numbers) :
    even_numbers = []
    for number in numbers:
        if number % 2 == 0: # filtering condition
            even_numbers.append(number)
    return even_numbers

extract_even(numbers)


