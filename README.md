# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv('titanic.csv')

# 1. Basic Information
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# 2. Missing Values Check
print(df.isnull().sum())

# Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

# 3. Check for Duplicates
print('Duplicate values:', df.duplicated().sum())

# 4. Univariate Analysis

# Categorical Features
categorical_features = ['Survived', 'Pclass', 'Sex', 'Embarked']
for feature in categorical_features:
    sns.countplot(x=feature, data=df)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Numerical Features
numerical_features = ['Age', 'Fare']
for feature in numerical_features:
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# 5. Bivariate Analysis

# Survival by Sex
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.show()

# Survival by Pclass
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()

# Age vs Survival
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age Distribution by Survival')
plt.show()

# Fare vs Survival
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare Distribution by Survival')
plt.show()

# 6. Multivariate Analysis
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# 7. Outlier Detection

# Boxplot for Age
sns.boxplot(x=df['Age'])
plt.title('Boxplot for Age')
plt.show()

# Boxplot for Fare
sns.boxplot(x=df['Fare'])
plt.title('Boxplot for Fare')
plt.show()

# 8. Feature Engineering Suggestions (Optional)
# Example: Creating Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Example: Creating Age Bins
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

# 9. Target Variable Distribution
sns.countplot(x='Survived', data=df)
plt.title('Overall Survival Distribution')
plt.show()

# 10. Final Dataset Overview
print(df.head())
print(df.info())
