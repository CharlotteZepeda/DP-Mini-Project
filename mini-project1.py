#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[4]:


import statsmodels


# In[8]:


ad = pd.read_csv("https://github.com/CharlotteZepeda/DP-Mini-Project/raw/380cff8680d5152a2d7e076ef1abd7f49c45073c/adni.csv")


# ### Data Checking

# In[23]:


ad.shape


# In[17]:


#check number of datapoints in each category 
ad.groupby("DIAGNOSIS").size()


# In[24]:


#check variable distributions
ad.hist(bins=30, figsize=(20,15))
plt.show()


# The distributions of the data show that there are: <br> 1851 people with an uneven distribution of the outcome (CN:Dementia:MCI = 617:348:886). 
# 

# In[9]:


ad.describe()
ad


# In[10]:


ad.info()


# In[11]:


# Import label encoder 
from sklearn import preprocessing
# label_encoder object knows how to understand   word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Smoking'. 
ad["Diagnosis_Ordinal"]= label_encoder.fit_transform(ad["DIAGNOSIS"]) 
print(ad.head())


# In[12]:


import scipy       as sp
import scipy.stats as stats
count = ad.count()

ad_stats = pd.DataFrame(count, columns = ["Non-NaN"])

min_vals = ad.min()
ad_stats['Min'] = min_vals

max_vals = ad.max()
ad_stats['Max'] = max_vals

mean_vals = ad.mean(skipna = True)
ad_stats['Mean'] = mean_vals

median_vals = ad.median(skipna=True)
ad_stats['Median'] = median_vals

ad_mode_sci = stats.mode(ad, nan_policy='omit')
ad_stats['Mode'] = ad_mode_sci[0][0]


ad_stats


# In[13]:


# Dropping variables with not enough data to reliably imputate values - all variables with more than 40% data missing were removed (threshold 60% NaN) 
ad_df = pd.DataFrame(ad)
ad_df = ad_df.drop(columns=['EcogPtMem.bl', 'EcogPtMem.bl', 'EcogPtLang.bl', 'EcogPtVisspat.bl', 'EcogPtPlan.bl','EcogPtOrgan.bl', 'EcogPtDivatt.bl', 'EcogPtTotal.bl', 'EcogSPMem.bl',
'EcogSPLang.bl','EcogSPVisspat.bl','EcogSPPlan.bl','EcogSPOrgan.bl','EcogSPDivatt.bl','EcogSPTotal.bl', 'MOCA.bl','PIB.bl','DIGITSCOR.bl', 'AV45.bl'])
ad_df.info()


# In[14]:


#crosstab for categorical variables 
data_crosstab1 = pd.crosstab(ad['PTGENDER'], 
                            ad['DIAGNOSIS']) 
print(data_crosstab1) 


# In[15]:


#replaced the 'Unknown' category with NAN and counted NANs across all the variables.

ad_df=ad_df.replace('Unknown', np.nan)
ad_df.isna().sum()


# In[22]:


# Wrote a simple for-loop to replace the function I wrote earlier duh
# I think these tables are a lot clearer

# Performing chi2 tests
from scipy.stats import chi2_contingency



cols = ['PTGENDER','PTETHCAT','PTRACCAT','PTMARRY']
for i in cols:
    ct = pd.crosstab(ad_df[i], ad_df['DIAGNOSIS'], margins=True)
    stat, pvalue, dof, expected = chi2_contingency(ct)
    print('\n', '\n', ct)
    print('Chi2 pvalue =', pvalue)
    



# ### Chi Squared Test Conclusions
# Chi Squared Test Results indicate : <br><br>
# (i) Significant association between Gender and Diagnosis Category **p value < .001** <br>
# (ii) Non-Significant association between ethnicity and Diagnosis Category **p value > 0.05** <br>
# (iii) Non-Significant association between Race and Diagnosis Category **p value > 0.05** <br>
# (iv) Significant association between Marital Status and Diagnosis Category **p value < .001**

# ### ANOVA Tests On Numerical Variables 
# 
# Basic Assumption of ANOVA is that **each group is drawn from a normal population**, therefore data should be rescaled (cell 24 shows not all variables are normally distributed)

# ### Column Rescaling
# 1. Rescale all the columns <br>
# 2. Make a deep copy <br>
# 3. Drop categorical variables from the deep copy<br>
# 4. Compute individual ANOVAs for each variable 

# In[53]:


# Rescale the columns of the dataframe
# for each value v in a distribution of values V:
#    v = (v-mean(V)) / std(V)

# Write a rescale() function


def rescale(ad_df):
    for col_name in ad_df.loc[:,:]:
        values = []
        mean_V = ad_df.loc[:,col_name].mean(skipna = True)
        std_V = ad_df.loc[:,col_name].std(skipna = True)
        for v in ad_df.loc[:, col_name]:
            v = (v - mean_V) / std_V
            values.append(v)
        ad_df[col_name] = values
    return ad_df


# In[54]:


# Make a deep copy of the data, holding the rescaled values:
cad_df = ad_df.copy()

# we want to rescale all numeric columns, but not diagnosis which is categorical
# use the drop function exclude some columns (axis=1) or rows (axis=0) from the dataframe (all categorical variables)
cad_dfdropped = cad_df.drop(['DIAGNOSIS','PTGENDER','PTETHCAT','PTRACCAT','PTMARRY'], axis=1)
cad_dfdropped


# In[55]:


# Now fill in cdf by applying the rescale function to each column except Outcome column
# then add again the categorical diagnosis column to cdf dataset

cad_df= rescale(cad_dfdropped)
cad_df['DIAGNOSIS']= ad_df['DIAGNOSIS']
cad_df



# In[56]:


cad_df['PTGENDER']= ad_df['PTGENDER']
cad_df


# In[57]:


cad_df['PTETHCAT']= ad_df['PTETHCAT']
cad_df


# In[58]:


cad_df['PTRACCAT']= ad_df['PTRACCAT']
cad_df


# ### Final Rescale Table Below
# (others can be deleted as this one contains all values) 

# In[60]:


cad_df['PTMARRY']= ad_df['PTMARRY']
cad_df


# In[61]:


g = sns.PairGrid(cad_df)
g.map(sns.scatterplot)


# In[64]:


#ANOVAS
#are they computed on ad_df values or cad_df values?
#(i) Age 

from scipy.stats import f_oneway

a1=cad_df[cad_df.DIAGNOSIS=='CN'].AGE
a2=cad_df[cad_df.DIAGNOSIS=='Dementia'].AGE
a3=cad_df[cad_df.DIAGNOSIS=='MCI'].AGE

a1 = a1[np.logical_not(np.isnan(a1))]
a2 = a2[np.logical_not(np.isnan(a2))]
a3 = a3[np.logical_not(np.isnan(a3))]
print(f_oneway(a1, a2, a3)


# In[82]:


#Loop for computing ANOVAS for numerical variables

cols = ['AGE','PTEDUCAT','ADAS11.bl','ADAS13.bl','ADASQ4.bl','MMSE.bl','RAVLT.immediate.bl','RAVLT.learning.bl','RAVLT.perc.forgetting.bl','LDELTOTAL.bl','TRABSCOR.bl','FAQ.bl','mPACCdigit.bl','mPACCtrailsB.bl','IMAGEUID.bl','Ventricles.bl','Hippocampus.bl','WholeBrain.bl','Entorhinal.bl','Fusiform.bl','MidTemp.bl','ICV.bl','ABETA.bl','TAU.bl','PTAU.bl','FDG.bl']
for i in cols:
    a1=cad_df[i][cad_df.DIAGNOSIS=='CN']
    a2=cad_df[i][cad_df.DIAGNOSIS=='Dementia']
    a3=cad_df[i][cad_df.DIAGNOSIS=='MCI']

    a1 = a1[np.logical_not(np.isnan(a1))]
    a2 = a2[np.logical_not(np.isnan(a2))]
    a3 = a3[np.logical_not(np.isnan(a3))]
    print(i) 
    print('Anova Test Result = ', f_oneway(a1, a2, a3))
    print('\n')


# In[80]:


#make a table of the results 


import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(header=dict(values=['Variable', 'ANOVA Result']),
                 cells=dict(values=[[Age, PTEDUCAT, ADAS11.bl, ADAS13.bl , ADASQ4.bl , MMSE.bl , RAVLT.immediate.bl , RAVLT.learning.bl , RAVLT.perc.forgetting.bl , LDELTOTAL.bl , TRABSCOR.bl , FAQ.bl, mPACCdigit.bl, mPACCtrailsB.bl, IMAGEUID.bl, Ventricles.bl, Hippocampus.bl, WholeBrain.bl, Entorhinal.bl, Fusiform.bl, MidTemp.bl, ICV.bl, ABETA.bl, TAU.bl, PTAU.bl, FDG.bl], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,]]))
                     ])
fig.show()


# ## ANOVA Test Conclusions
# 

# ### Data Imputation 

# In[ ]:




