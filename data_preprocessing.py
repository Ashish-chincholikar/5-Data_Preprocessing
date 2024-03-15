# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:31:38 2023

@author: 91721

Data Preprocessing
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Data_Set/ethnic diversity.csv")
df

#step1 -  perform the operations related to EDA 
#how many data-points and features
df.shape
#(310, 13)

#What are the Column Names in our dataset
df.columns

#datatypes
df.dtypes

#converting datatypes
#salaries is in float , converting it into integer type
df.Salaries = df.Salaries.astype(int)
df.dtypes

#age is in integer , converting it into float type 
df.age = df.age.astype(float)
df.dtypes
#How many Data points for each class in the department type
df["Department"].value_counts()

"""
Production              208
IT/IS                    50
Sales                    31
Admin Offices            10
Software Engineering     10
Executive Office          1
Name: Department, dtype: int64
 """
 
#2-D Scatter plot
df.plot(kind = "scatter" , y ="age" , x="Department" )
plt.show()

#---------------------------------------------------------------
#identifying the duplicates-->and then drop it

df_new = pd.read_csv("C:/Data_Set/education.csv")
duplicate = df_new.duplicated()
duplicate
#duplicated() method output is single column
#if duplicted found output is true, if duplicates not found the output is false
sum(duplicate)

df_new1 = pd.read_csv("C:/Data_Set/mtcars_dup.csv")
duplicate1 = df_new1.duplicated()
duplicate1
"""3 duplicate records found at 17 , 23 and 27 index
we need to drop these records
row 17 is duplicate of 2
"""
sum(duplicate1)
"""in order to drop duplicates we have dropduplicates() method """

df_new2 = df_new1.drop_duplicates()
df_new2
duplicate3 = df_new2.duplicated()
sum(duplicate3)
#-----------------

#outlier treatment
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Data_Set/ethnic diversity.csv")
df

sns.boxplot(df.Salaries)
#there are outliers 
#let us check outliers in age columns

sns.boxplot(df.age)
#There are no Outliers 
#let us check IQR

IQR = df.Salaries.quantile(0.75) - df.Salaries.quantile(0.25)
#IQR = Q3-Q1 
IQR

#Now we define the higher and lower limit so as to 
#take a decision for the datapoints that must be treated
#as outliers

lower_limit = df.Salaries.quantile(0.25)-1.5*IQR
higher_limit = df.Salaries.quantile(0.75) + 1.5*IQR

#here the lower limit was in -ve make it zero 
#in variable explorer using the lower_limit variable

#---------------------------------------
#trimming
import numpy as np 
outliers_df = np.where(df.Salaries>higher_limit , True , np.where(df.Salaries<lower_limit , True ,False))

#you can check outlier_df option in Variable explorer
#here 4 outliers are found at the index 23 and so on an outlier is found 
df_trimmed = df.loc[~outliers_df]

#check the shape of original dataframe
df.shape
#(310,13)

#after trimming
df_trimmed.shape
#(306 , 13)

sns.boxplot(df_trimmed.Salaries)
#-----------------------------------------------------
#masking and replacment technique

df = pd.read_csv("C:/Data_Set/ethnic diversity.csv")
df.describe()

#record 23 has an outlier
#map all the outlier value to the upper limit
df_replaced = pd.DataFrame(np.where(df.Salaries>higher_limit , higher_limit , np.where(df.Salaries<lower_limit , lower_limit , df.Salaries)))

#if the values are greater than upper_limit
#map to upper_limit 
#if the values are less than lower limit 
#map to lower_limit

sns.boxplot(df_replaced[0])
"""
Created on Fri Oct  6 08:11:28 2023

@author: 91721
"""

import pandas as pd

df_new = pd.read_csv("C:/Data_Set/OnlineRetail.csv")

#------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Data_Set/ethnic diversity.csv")

#Winsorizer
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method='iqr' , tail = 'both',fold = 1.5 , variables=['Salaries'])

#copy winsorizer and paste in help tab of
#top right window , study the method

df_t = winsor.fit_transform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])
#--------------------------------------------------
#practise on boston data set

"""

    
The Boston Housing Dataset

The Boston Housing Dataset is a derived from information collected by the U.S.
Census Service concerning housing in the area of Boston MA.

It has two prototasks: nox, in which the nitrous oxide level is to be predicted; 
and price, in which the median value of a home is to be predicted

The following describes the dataset columns:
There are 14 attributes in each case of the dataset

CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS - proportion of non-retail business acres per town.
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX - nitric oxides concentration (parts per 10 million)
RM - average number of rooms per dwelling
AGE - proportion of owner-occupied units built prior to 1940
DIS - weighted distances to five Boston employment centres
RAD - index of accessibility to radial highways
TAX - full-value property-tax rate per $10,000
PTRATIO - pupil-teacher ratio by town
B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT - % lower status of the population
MEDV - Median value of owner-occupied homes in $1000's

Business objective:

Maximize: the prediction of nitrous oxide level, 
and the price of median value of home. (in boston) 
    
Minimize: 

constraints : Data authenticity and effective collection means

Challenges:

Missing value treatment
Outlier treatment
Understanding which variables drive the price of homes in Boston

data dictionary : 
 
EDA

Data preprocessing

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Data_Set/Boston.csv")

#shape of boston dataset
df.shape
#(506, 15)

#identify the DataTypes
df.dtypes
""" 
Unnamed: 0      int64
crim          float64
zn            float64
indus         float64
chas            int64
nox           float64
rm            float64
age           float64
dis           float64
rad             int64
tax             int64
ptratio       float64
black         float64
lstat         float64
medv          float64
dtype: object

here all the Columns datatypes are mostly correct . Ratio , interval , Discrete
Datatypes can be observerd."""

df.columns
df.describe()

#changing the datatype of RM to int 
#because the number of rooms for dwelling must be integer
df.rm = df.rm.astype(int)

#2-D Scatter plot
df.plot(kind="scatter" , x = 'rm' , y='nox' )
plt.show
"""inferense from the scatter plot for number of rooms for dwelling vs
the nitrogen oxides level 
1. """
df["nox"].value_counts()

#RM , NOX


"""
Created on Mon Oct  9 08:33:38 2023

@author: 91721
"""

""" zero variance and near zero variance 
if there is no variance in the feature , then ML model will not get intelligent
, so it is better to ignore those features"""

import pandas as pd
df = pd.read_csv("C:/Data_Set/ethnic diversity.csv")
df.var()
#here EmpId and ZIP is nominal data
#salaries has 4.441953e+08 is 4441953000 which is not close to 0
#similarly age 8.571358e+01 = 85.71
#both the features are having considerable variance

df.var()==0
"""EmpID       False
Zip         False
Salaries    False
age         False
dtype: bool """

#none of them are equal to 0
df.var(axis=0)==0
"""axis = 0 is for rows
EmpID       False
Zip         False
Salaries    False
age         False
dtype: bool """
#-----------------------------------------------------------------
#data_preprosing file in jyupter notebook for missing values imputation
import pandas as pd
import numpy as np
df = pd.read_csv("C:/Data_Set/modified ethnic.csv")
df
#Check for Null Values
df.isna().sum()
"""Out[24]: 
Position            43
State               35
Sex                 34
MaritalDesc         29
CitizenDesc         27
EmploymentStatus    32
Department          18
Salaries            32
age                 35
Race                25
dtype: int64
here 43 missing values are find in the Position columns """

#create an imputer that creates NaN values
#mean and median is used for numeric data
#mode is used for discrete data(position , sex , MaritalDes)
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values = np.nan , strategy="mean")
#Check the DataFrame
df['Salaries'] = pd.DataFrame(mean_imputer.fit_transform(df[['Salaries']]))
#check the dataFrame
df['Salaries'].isna().sum()
#Out[32]: 0

#--------------------------------------------------------------------
import pandas as pd

data = pd.read_csv("C:/Data_Set/ethnic diversity.csv")
data.head(10)#-->shows first 10 records of the dataset
data.info()
#it gives size, null values , rows , columns and columns datatypes

data.describe()
data['Salaries_new'] = pd.cut(data['Salaries'],bins=[min(data.Salaries),data.Salaries.mean() ,max(data.Salaries)] , labels=['high' , 'low'])
data.Salaries_new.value_counts()
data['Salaries_new'] = pd.cut(data['Salaries'] , bins = [min(data.Salaries),data.Salaries.quantile(0.25),data.Salaries.mean() , data.Salaries.quantile(0.75),max(data.Salaries)],labels = ["group1" ,"group2" , "group3" , "group4"])
data.Salaries_new.value_counts()

#----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Data_Set/animal_category.csv")
df.shape
df.drop(['Index'],axis = 1 , inplace = True)
#check df again
df

df_new = pd.get_dummies(df)
df_new.shape

#here we are getting 30 rows and 14 columns
#we are getting two columns for homely and gender , one column
#delete second column of gender and second column of homely
df_new.drop(['Gender_Male' , 'Homly_Yes'] , axis=1 , inplace=True)
df_new.shape
#Now we are getting 30,12
df_new.rename(columns = {'Gender_Female':'Gender' , 'homly_no':'house_no'})
#-------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Data_Set/ethnic diversity.csv")
df.shape

df_new = pd.get_dummies(df)
df_new.shape


df_new.drop(['Sex_F' , 'CitizenDesc_Non-Citizen'] , axis=1 , inplace=True)
df_new.columns
df_new.shape
df["Sex"].value_counts()
df["CitizenDesc"].value_count()
df_new.rename(columns = {'Sex_F':'SEX' , 'CitizenDesc_Citizen' : 'CITIZEN'})
"""
Created on Tue Oct 10 08:25:36 2023

@author: 91721
"""

#techniques of Creation of Dummy Variables
#OneHotEncoder()

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
#we use enthnic diversity dataset

df = pd.read_csv("C:/Data_Set/ethnic diversity.csv")
df.columns
#We Have Salaries and age as numerical columns , let us make 
#them at position 0 and 1 so to make further data processing easy

df = df[['Salaries','age' , 'Sex','Employee_Name', 'EmpID', 'Position', 'State', 'Zip',
       'MaritalDesc', 'CitizenDesc', 'EmploymentStatus', 'Department','Race']]
#Check the dataframe in variable explorer
#we want only nominal Data and ordinal data for processing
#hence skipped 0th and 1st column and applied to one hot Encoder

enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:,2:]).toarray())
#OneHotEncoding() losses the column labels that is the drawback of this technique


#label Encoder
from sklearn.preprocessing import LabelEncoder

#creating instance of Label encoder
labelencoder = LabelEncoder()
#split your data into input and output variable
X = df.iloc[: , 0:9]#first eight columns for X and 9th for y
y = df['Race']
df.columns

#we have nominal data Sex , MaritalDesc , CitizenDesc
#we want to convert to label Encoder

X['Sex'] = labelencoder.fit_transform(X['Sex'])
X['MaritalDesc'] = labelencoder.fit_transform(X['MaritalDesc'])
X['CitizenDesc'] = labelencoder.fit_transform(X['CitizenDesc'])

#label encoder y
y = labelencoder.fit_transform(y)
#this is going to create an array , hence convert
#it back to dataframe
y = pd.DataFrame(y)
df_new = pd.concat([X,y] , axis=1)

#if you will see variable explorer , y do not have column name
#hence rename the column

df_new = df_new.rename(columns={0:'Race'})

#normalization and standardilization

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

d = pd.read_csv("C:/Data_Set/mtcars_dup.csv")
a = d.describe()

#here the mean is very unevenly distributed

#Initialize the Scaler
scaler = StandardScaler()
df = scaler.fit_transform(d)
dataset = pd.DataFrame(df)
res = dataset.describe()
#-------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/Data_Set/Seeds_data.csv")
des = d.describe()

#here the mean and standard-deviation is very unevenly distributed
scaler = StandardScaler()
df_new = scaler.fit_transform(df)
dataset = pd.DataFrame(df_new)
res1 = dataset.describe()

#inference : we obtain the standard deviation that is equally distributed
#among all the features of the dataset
#-------------------------------------------------------------------------
#Normalization

ethnic = pd.read_csv("C:/Data_Set/ethnic diversity.csv")
#now read columns
ethnic.columns
#there are some columns which are not useful , we need to drop
ethnic.drop(['Employee_Name' , 'EmpID' ,'Zip'] , axis=1 , inplace = True)
#now read minimum value and maximum values of Salaries and age
a1 = ethnic.describe()
#check a1 data frame in variable explorer.
#you find minimum salary is 0 and max is 108304
#same way check for age , there is huge difference
#in min and max values. hence we are going for normalization
#first we will have to convert non-numeric data to label encoding
ethnic = pd.get_dummies(ethnic , drop_first=True)
#Normalization function written where ethnic argument is passed
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm = norm_func(ethnic)
b = df_norm.describe()

#if you will observe the b frame,
#it has dimensations 8,81
#earlier in a they were 8,11, it is because all non-numeric
#data had been connverted to numeric using label encoding

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:42:32 2023

@author: 91721
"""
import pandas as pd
import matplotlib.pyplot as plt
#Now import file from data set and create a dataFrame
Univ1 = pd.read_excel("C:/Data_Set/University_Clustering.xlsx")
a = Univ1.describe()
#We have one column "State" which really not useful we will drop it
Univ = Univ1.drop(["State"] , axis=1)
#We know that there is scale difference among the columns,
#which we have to remove
#either by using normalization or standardization
#Whenever there is mixed data apply normalization

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#Now apply this normalization fucntion to Univ dataFrame for all the rows
#Since 0th column has University name hence skipped
df_norm = norm_func(Univ.iloc[:,1:])
#you can ckeck the df_norm dataFrame which is scaled between 
#values of 0 and 1
#you can apply describe function to new data frame 
b = df_norm.describe()
#Before you apply clustering , you need to plot dendogram first
#Now to create dendogram , we need to measure distance,
#we have to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierarchicla or aglomerative clustering 
#ref the help for linkage
z = linkage(df_norm , method="complete" , metric="euclidean")
plt.figure(figsize = (15,8));
plt.title("Hierarchical Clustering dendogram");
plt.xlabel("Index");
plt.ylabel("Distance")
#ref help of dendogram
#sch.dendogram(z)
sch.dendrogram(z,leaf_rotation=0, leaf_font_size =10)
plt.show()
#dendogram()
#applying agglomerative clustering choosing 3 as clusters
#from dendogram
#whatever has been displayed is dendogram is not clustering
#it is just showing numbers of possible clusters

from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=3 , linkage="complete",affinity="euclidean").fit(df_norm)

#apply labels to the clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#Assign this series to Univ DataFrame as columns and name the columsbn
Univ['clust'] = cluster_labels
#we want to relocate the columns 7 to 0 th position
Univ1.shape
Univ1 = Univ.iloc[: , [7,1,2,3,4,5,6]]
#now check the Univ1 dataframe
Univ = Univ.iloc[:,2:].groupby(Univ1.clust).mean()
#from the output clusters 2 has got higest Top10
#lowest accept ratio, best faculty ration and higest expenses
#higest graduates ratio
Univ1.to_csv("C:/4-DataPreprosesing/University.csv" , encoding="utf8")
import os
os.getcwd()

