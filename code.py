import pandas as pd
import numpy as np

df=pd.read_csv("data/online_retail.csv")

df=df.dropna()

df

df.isnull().sum()

df.dtypes

df.info()

df.describe()

df.value_counts(subset=['Description','Country'])

# Remove negative quantities (returns/refunds)
df = df[df['Quantity'] > 0]

# Convert InvoiceDate to proper datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create TotalPrice column (Revenue)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Final check
print(df.head())
print(df.info())

print(df['Quantity'].min())   # should be > 0
print(df.isnull().sum())      # should be 0

import matplotlib.pyplot as plt
import seaborn as sns


# Create snapshot date (latest date in dataset)
snapshot_date = df['InvoiceDate'].max()

# Group by CustomerID and calculate RFM
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalPrice': 'sum'  # Monetary
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Reset index (optional but better)
rfm = rfm.reset_index()

# Show result
print(rfm.head())

# RFM Scoring (1–4 scale)

rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])

# Convert to int
rfm['R_score'] = rfm['R_score'].astype(int)
rfm['F_score'] = rfm['F_score'].astype(int)
rfm['M_score'] = rfm['M_score'].astype(int)

# Total RFM Score
rfm['RFM_Score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

def segment_customer(row):
    if row['RFM_Score'] >= 10:
        return 'VIP Customer'
    elif row['RFM_Score'] >= 7:
        return 'Loyal Customer'
    elif row['RFM_Score'] >= 5:
        return 'Potential Customer'
    else:
        return 'At Risk Customer'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# Show result
print(rfm.head())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.countplot(data=rfm, x='Segment')
plt.title("Customer Segments Distribution")
plt.xticks(rotation=30)
plt.show()

# - VIP customers contribute the highest revenue.
# - Customers with low recency are more active.
# - High frequency customers are more loyal.
# - At-risk customers need re-engagement strategies.

sns.boxplot(x='Segment', y='Monetary', data=rfm)
plt.title("Monetary Value by Segment")
plt.show()

rfm.to_csv("rfm_final.csv", index=False)

