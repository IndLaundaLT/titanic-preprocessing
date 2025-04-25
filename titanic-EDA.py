# Titanic EDA - Task 2

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\coding\Python\titanic-preprocessing\titanic.csv')

# Initial look at the dataset
print(df.head())
print(df.info())
print(df.describe(include='all'))
print("\nMissing values:\n", df.isnull().sum())

# Fill missing values (if any)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)

# Summary statistics
print("\nValue counts for categorical features:\n")
print(df['Sex'].value_counts())
print(df['Pclass'].value_counts())

# Univariate Analysis
plt.figure(figsize=(10, 4))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age by Passenger Class')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

# Bivariate Analysis
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Sex')
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True)
plt.title('Age vs Survival')
plt.show()

# Multivariate Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']])
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()

# Conclusions:
print("""
Key Insights:
- Females had a higher survival rate than males.
- Passengers in 1st class had a better survival rate.
- Children and younger passengers had slightly higher survival odds.
- Strong correlation between Pclass and Fare.
""")