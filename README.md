# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
# STEP 1:
Read the given Data

# STEP 2:
Clean the Data Set using Data Cleaning Process

# STEP 3:
Apply Feature Transformation techniques to all the features of the data set

# STEP 4:
Print the transformed features

# PPROGRAM:
```
Name:M.D. Harini 
Reg No. 212222230043


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

# READ CSV FILES
df=pd.read_csv("/content/Data_to_Transform.csv")
df
# BASIC PROCESS
df.head()

df.info()

df.describe()

df.tail()

df.shape

df.columns

df.isnull().sum()

df.duplicated()

# LOG TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# MODERATE POSITIVE SKEW
df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

# RECIPROCAL TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# SQUARE ROOT TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# POWER TRANSFORMATION
# MODERATE POSITIVE SKEW
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

# MODERATE NEGATIVE SKEW
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")

df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

# QUANTILE TRANSFORMATION
# MODERATE NEGATIVE SKEW
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')

df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
# OUTPUT:
# Importing Libraries
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/ba4ac67f-0855-460a-85f1-699e0084ec64)
# Reading CSV File
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/2afb767e-1aa4-4133-9a04-1da302a48605)
# Basic Process
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/66c5d8fa-4248-4c8b-ab21-1311f3f348cc)
# Before Transformation
## Highly Positive Skew
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/12932b65-1f00-4cb9-9fd8-f24c09cad929)
## Highly Negative Skew
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/e632c937-91f7-47cf-89b1-a8b28c7f03e4)
## Moderate Positive Skew
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/2dc266b0-562a-4c58-b652-ef88079ccbe0)
## Moderate Negative Skew
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/4e034dbf-0bc6-42cc-8382-558aaa69a678)
# Log Transformation
## Highly Positive Skew
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/f060e11b-14b5-4b15-8de0-8188dd926034)
## Moderate Positive Skew
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/fe2d60f9-3626-46b2-9966-f5838a23e1cc)
# Reciprocal Transformation
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/5be1a5fd-9294-42c9-9840-7a953a56ac0c)
# Square Root Transformation
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/d2e03280-c744-4a71-b8bb-12d0cd8e9885)
# Power Transformation
## Moderate Positive Skew
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/3e6b61c4-57ac-4485-90d2-a3d8c642e1af)
## Moderate Negative Skew
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/a19f481b-b8c0-4465-85a1-d58469c33e38)
# Quantile Transformation
## Moderate Negative Skew
![image](https://github.com/harinidq/Ex-06-Feature-Transformation/assets/113497680/60cfd963-e151-4484-b762-266c706e4164)

# RESULT:
Thus feature transformation is done for the given dataset.
