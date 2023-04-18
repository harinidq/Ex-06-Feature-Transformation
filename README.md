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

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
# OUTPUT:
![image](https://user-images.githubusercontent.com/113497680/232675149-7437bd54-1fae-42f8-bea4-08750aa70b9c.png)

![image](https://user-images.githubusercontent.com/113497680/232675250-4a7bcce2-968b-41e8-89ec-3d5f9e407138.png)

![image](https://user-images.githubusercontent.com/113497680/232675286-027e6afd-7ec0-41d4-ba52-d40e64f477a2.png)

![image](https://user-images.githubusercontent.com/113497680/232675322-73eeb783-767b-4adb-931c-7a0e274d777e.png)

![image](https://user-images.githubusercontent.com/113497680/232675354-433d774e-0e35-4c11-aafa-c250d7ec852f.png)

![image](https://user-images.githubusercontent.com/113497680/232675395-3f1302d2-a60a-4429-ae4d-20b4874f0a7e.png)

![image](https://user-images.githubusercontent.com/113497680/232675432-6a96a00a-d253-45d3-b1f3-8a40dd10f02f.png)

![image](https://user-images.githubusercontent.com/113497680/232675473-a25df458-3585-4dd1-bcf2-123de2599052.png)

![image](https://user-images.githubusercontent.com/113497680/232675498-de4fd801-774b-4aaa-bd79-439616a1ef8c.png)

![image](https://user-images.githubusercontent.com/113497680/232675519-0adfe302-30b8-422a-91d5-7fe71df40fbd.png)

# RESULT:
Thus feature transformation is done for the given dataset.
