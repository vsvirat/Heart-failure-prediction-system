import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#KAGGLE DATASET
df = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
df

(df.columns)   # printing features 

import seaborn as sns
sns.pairplot(df)           

df["DEATH_EVENT"].value_counts()          # counting total deaths and alive from deathcount column

df["DEATH_EVENT"].value_counts().plot(kind="bar", color=["salmon", "lightblue"]);
plt.xlabel("DEATH COUNT")
plt.ylabel("COUNT")

df.isna().sum()  # checking for null values

df.describe()

# HEART FAILURE FREQUENCY ON THE BASICS OF GENDER
df.rename(columns = {"sex": "gender"} , inplace= True)
df["gender"].value_counts()

#crosstab values 
pd.crosstab(df.DEATH_EVENT , df.gender)

pd.crosstab(df.DEATH_EVENT , df.gender).plot( kind="bar" , figsize=(10,6) , color=["salmon" , "lightblue"] )
plt.title("Heart Disease Frequency for Gender")
plt.xlabel("0= No Death        |          1= Death")
plt.ylabel("Count")
plt.legend( ["Female", "Male"] )

#HEART FAILURE FREQUENCY ON THE BASICS OF AGE
df["age"].value_counts().head()

#crosstab values 
pd.crosstab(df.DEATH_EVENT , df.age)

corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim()
