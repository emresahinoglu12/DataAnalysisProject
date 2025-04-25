import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

if not os.path.exists("visuals"):
    os.makedirs("visuals")

data = pd.read_csv('data/cinema_hall_ticket_sales.csv')

# Data Cleaning
# -------------------

data['Number_of_Person'] = data['Number_of_Person'].replace('Alone', 1).astype(int)

# Map Yes/No
data['Purchase_Again'] = data['Purchase_Again'].map({'Yes': 1, 'No': 0})

# -------------------
# Quick Data Overview

print(data.info())
print(data.describe())
print(data.head())


# Exploratory Data Analysis (EDA)
# -------------------

# For Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('visuals/age_distribution.png')
plt.close()

# For Ticket Price vs Seat Type
plt.figure(figsize=(8,5))
sns.boxplot(x='Seat_Type', y='Ticket_Price', data=data)
plt.title('Ticket Price by Seat Type')
plt.savefig('visuals/ticket_price_seat_type.png')
plt.close()

# For Movie Genre Popularity
plt.figure(figsize=(8,5))
sns.countplot(x='Movie_Genre', data=data, order=data['Movie_Genre'].value_counts().index)
plt.title('Movie Genre Popularity')
plt.ylabel('Number of Tickets Sold')
plt.savefig('visuals/genre_popularity.png')
plt.close()

# For Purchase Again Rate by Age Group
data['Age_Group'] = pd.cut(data['Age'], bins=[0, 17, 35, 50, 65, 100], labels=['<18','18-35','36-50','51-65','65+'])
purchase_rate = data.groupby('Age_Group', observed=False)['Purchase_Again'].mean()

purchase_rate.plot(kind='bar', figsize=(8,5), title='Repeat Purchase Rate by Age Group', color='skyblue')
plt.ylabel('Repeat Purchase Rate')
plt.savefig('visuals/repeat_purchase_age_group.png')
plt.close()

# For Correlation Heatmap (Numeric Columns Only)
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('visuals/correlation_matrix.png')
plt.close()


# Hypothesis Testing
# -------------------

# - Older audiences >40 prefer Premium seating
data['Older'] = data['Age'] > 40
premium_pref = pd.crosstab(data['Older'], data['Seat_Type'] == 'Premium')

chi2, p, dof, ex = stats.chi2_contingency(premium_pref)
print(f"\n[Hypothesis 1] Chi-Square Test for Premium Seat Preference (Older vs Younger): p-value = {p:.4f}")

# - Younger audiences (18-35) visits more
young_group = data[data['Age_Group'] == '18-35']['Purchase_Again']
other_groups = data[data['Age_Group'] != '18-35']['Purchase_Again']

t_stat, p_value = stats.ttest_ind(young_group, other_groups)
print(f"[Hypothesis 2] T-Test for Repeat Purchase (18-35 vs Others): p-value = {p_value:.4f}")


# Save Cleaned Data
# -------------------
data.to_csv('data/cleaned_cinema_data.csv', index=False)

print("\n✅ EDA and Hypothesis Testing Completed. Visuals saved in /visuals folder.")
