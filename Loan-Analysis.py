#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/seerat16/loan-risk-analysis/main/loan%20data.csv')
df.shape


# In[3]:


df.head()


# In[4]:


df['TARGET'].value_counts()


# In[5]:


missing_val = df.isnull().sum()
missing_val


# In[6]:


missing_percentage = pd.DataFrame({'Columns':missing_val.index,'Percentage':(missing_val.values/df.shape[0])*100})
missing_percentage


# In[7]:


import seaborn as sns


# In[8]:


plt.figure(figsize=(20,7))
sns.pointplot(data=missing_percentage,x='Columns',y='Percentage')
plt.axhline(50,color='r',linestyle='--')
plt.title('Missing % in the Dataset',fontsize=30)
plt.xticks(rotation=90)
plt.show()




# In[9]:


missing_more_50 = missing_percentage[missing_percentage['Percentage']>=50]
missing_more_50


# In[10]:


df1=df.drop(columns=missing_more_50.Columns)

df1.shape
df1.head()


# In[11]:


columns_to_delete= ['FLAG_DOCUMENT_2',
       'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
       'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
       'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
       'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
       'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
       'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
       'FLAG_DOCUMENT_21','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE',
       'FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','DAYS_LAST_PHONE_CHANGE',
       'REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION',
       'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','EXT_SOURCE_2',
       'EXT_SOURCE_3','REGION_RATING_CLIENT_W_CITY','YEARS_BEGINEXPLUATATION_AVG','FLOORSMAX_AVG',
       'YEARS_BEGINEXPLUATATION_MODE','FLOORSMAX_MODE','YEARS_BEGINEXPLUATATION_MEDI',
       'FLOORSMAX_MEDI','TOTALAREA_MODE','EMERGENCYSTATE_MODE']

df2= df1.drop(columns=columns_to_delete)
df2.shape



# In[12]:


df2.head()


# In[13]:


less_50_missing = (df2.isnull().sum()/df2.shape[0])*100
less_50_missing[less_50_missing>0]


# In[14]:


df2.AMT_GOODS_PRICE.describe()


# In[15]:


df2['AMT_GOODS_PRICE']=df2['AMT_GOODS_PRICE'].fillna(df2['AMT_GOODS_PRICE'].median())


# In[16]:


df2.NAME_TYPE_SUITE.value_counts()


# In[17]:


df2['NAME_TYPE_SUITE']=df2['NAME_TYPE_SUITE'].fillna('Unaccompanied')


# In[18]:


df2.SK_ID_CURR.nunique()


# In[19]:


df2.describe()


# In[20]:


df2.DAYS_BIRTH = df2.DAYS_BIRTH.abs()
df2.DAYS_EMPLOYED = df2.DAYS_EMPLOYED.abs()
df2.DAYS_REGISTRATION = df2.DAYS_REGISTRATION.abs()
df2.DAYS_ID_PUBLISH = df2.DAYS_ID_PUBLISH.abs()


# In[21]:


df2['Age_Range'] = (df2.DAYS_BIRTH / 365).round(2)
df2.head()


# In[22]:


bins = [0,30,40,50,60,100]
labels = ['<30','30-40','40-50','50-60','60+']
df2['AGE_RANGE'] = pd.cut(df2.Age_Range,bins=bins,labels=labels)
df2.drop(columns='Age_Range',inplace=True)


# In[23]:


df2.TARGET.value_counts(normalize=True)*100


# In[24]:


fig, (ax1,ax2) = plt.subplots(1,2,figsize =(20,8))

ax = sns.countplot(df2.TARGET,ax=ax1)

ax1.set_title('TARGET',fontsize=20)

plt.setp(ax1.xaxis.get_majorticklabels(),fontsize=18)

ax2 = plt.pie(x=df2.TARGET.value_counts(normalize=True),autopct='%.2f',textprops={'fontsize':15},shadow=True,labels=['No Payment Issues','Payment Issues'],wedgeprops = {'linewidth': 5}) 

plt.title('Distribution of the Target Variable',fontsize=20)

plt.show()


# In[25]:


sns.set_style(style = 'whitegrid',rc={"grid.linewidth": 5})


# In[26]:


df2.nunique().sort_values()


# In[27]:


# Function for univariate analysis
def plots(l,rows=1,cols=1,rot=90):
        
    if cols>1:
        fig, (ax1,ax2) = plt.subplots(nrows=rows,ncols=cols,figsize=(30,10))
        fig.subplots_adjust(hspace = .2, wspace=.2)
    
    else:
        fig, (ax1,ax2) = plt.subplots(nrows=rows,ncols=cols,figsize=(30,30))
        fig.subplots_adjust(hspace = .5, wspace=.1)
    
    
    # Subplot 1 : countplot 
    first = sns.countplot(data = df2 , hue = 'TARGET', palette='inferno',x=l,ax=ax1)
    first.set_title(l,fontsize=30)
    first.set_yscale('log')
    first.legend(labels=['Loan Repayers','Loan Defaulters'],fontsize=20)
    plt.setp(first.xaxis.get_majorticklabels(), rotation=rot,fontsize=25)
    plt.setp(first.yaxis.get_majorticklabels(),fontsize=18)


    # Percentage of the mean values for defaulters
    default_percentage = (df2.groupby(by=l)['TARGET'].mean()*100).sort_values()
    
     # Subplot 2 : barplot
    sec = sns.barplot(x=default_percentage.index,y=default_percentage,ax=ax2)
    sec.set_title(f'Default % in {l}',fontsize=30)
    sec.set_yscale('linear')
    plt.setp(sec.xaxis.get_majorticklabels(), rotation=rot,fontsize=25)
    plt.setp(sec.yaxis.get_majorticklabels(),fontsize=18)
    return None


# In[28]:


list_categories = ['FLAG_OWN_CAR','FLAG_OWN_REALTY','CODE_GENDER','NAME_CONTRACT_TYPE']

for val in list_categories:
    plots(val,1,2,rot=0)


# In[29]:


list_categories = ['NAME_INCOME_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE']

for i in list_categories:
    plots(i,1,2,rot=50)


# In[30]:


plots('OCCUPATION_TYPE',1,2)


# In[31]:


plots('ORGANIZATION_TYPE',rows=2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




