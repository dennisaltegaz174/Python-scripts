import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
laterite = pd.read_csv("C:/Users/adm/Documents/Datasets/laterite_mobilemoney_data.csv")
laterite.head()
# 1 data to have one observation per participant
laterite=laterite.sort_values(by="hhid")
laterite.drop_duplicates('hhid',keep='first')

# Creating dummy variables
laterite['financially_excluded'] = np.where(laterite['account_type']== 'None',1,0)
laterite.head()
laterite['digitally_financially_included']=np.where(laterite['account_type']=='None',0,1)

# Overall Rate of  financial exclusion and  digital financial inclusion for combined pop in 3  districts
g=sns.catplot(x='district',hue='financially_excluded',
           kind='bar',data=laterite)



fig=plt.figure(figsize=(18,15))
ax1=plt.subplot2grid((1,2),(0,0))
plt.hist(laterite ,weights=np.ones(len(laterite))/len(laterite))
sns.countplot(data=laterite , x='district',weights=np.ones(len(laterite))/len(laterite),hue='financially_excluded',palette='viridis')
plt.title("RATE OF FINANCIAL EXCLUSION IN THE \n THREE DISTRICTS",fontsize=10,weight='bold')

ax1=plt.subplot2grid((1,2),(0,1))
sns.countplot(data=laterite , x='digitally_financially_included',hue='district',palette='crest')
plt.title("RATE OF DIGITAL FINANCIAL INCLUSION \n IN THE THREE DISTRICTS",fontsize=10,weight='bold')


# Good
g=sns.factorplot('district',data=laterite,kind='count',hue='digitally_financially_included',palette='viridis',aspect=1)
g.ax.set_ylim(0,100)
for p in g.ax.patches:
    txt=
str(p.get_height().round(1))
    + "%"
    txt_x=p.get_x()
    txt_y=p.get_height()
