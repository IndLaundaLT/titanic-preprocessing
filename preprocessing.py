# Titanic Dataset - Data Cleaning & Preprocessing

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Step 2: Load dataset
df = pd.read_csv("titanic.csv")

# Step 3: Basic Exploration
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Step 4: Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns='Cabin', inplace=True)  # Optional, many missing values

# Step 5: Encode Categorical Variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Step 6: Feature Scaling
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Step 7: Outlier Detection & Removal
sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot")
plt.show()

# Remove outliers
fare_threshold = 300
df = df[df['Fare'] < fare_threshold]

# Save cleaned data
df.to_csv("titanic_cleaned.csv", index=False)
print("Data cleaned and saved as 'titanic_cleaned.csv'")
