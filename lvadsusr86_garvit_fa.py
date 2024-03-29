# -*- coding: utf-8 -*-
"""LVADSUSR86-garvit-FA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yVU50w0Vtp83ytoiIvaeWFlxb5JgUOOL
"""

import pandas as pd
import numpy as np

#1

data = pd.read_excel('/content/Walmart_Dataset Python_Final_Assessment.xlsx')

print(data.info())

print(data.describe())

data.head(10)

# 2

print(data.isnull().sum())

#cheackin for any duplicate entries
print("Duplicate entries:", data.duplicated().sum())
# Dropping any duplicate entries that are preasent
data.drop_duplicates()

#filling value tho be zero where there is a null value in the dataset
data.fillna(0, inplace=True)

# Here i am handling whitespaces or null values with nan
data.replace(r'^\s*$', float('nan'), regex=True, inplace=True)

#3

sales_data = data['Sales']

sales_mean = np.mean(sales_data)
sales_median = np.median(sales_data)
sales_mode = sales_data.mode()[0]
sales_range = sales_data.max() - sales_data.min()
sales_variance = np.var(sales_data)
sales_std_dev = np.std(sales_data)

print("Mean sales Level:", sales_mean)
print("Median sales Level:", sales_median)
print("Mode sales Level:", sales_mode)
print("Range of sales Levels:", sales_range)
print("Variance of sales Levels:", sales_variance)
print("Standard Deviation of sales Levels:", sales_std_dev)

Profit_data = data['Profit']

Profit_mean = np.mean(Profit_data)
Profit_median = np.median(Profit_data)
Profit_mode = Profit_data.mode()[0]
Profit_range = Profit_data.max() - Profit_data.min()
Profit_variance = np.var(Profit_data)
Profit_std_dev = np.std(Profit_data)

print("Mean Profit Level:", Profit_mean)
print("Median Profit Level:", Profit_median)
print("Mode Profit Level:", Profit_mode)
print("Range of Profit Levels:", Profit_range)
print("Variance of Profit Levels:", Profit_variance)
print("Standard Deviation of Profit Levels:", Profit_std_dev)

Quantity_data = data['Quantity']

Quantity_mean = np.mean(Quantity_data)
Quantity_median = np.median(Quantity_data)
Quantity_mode = Quantity_data.mode()[0]
Quantity_range = Quantity_data.max() - Quantity_data.min()
Quantity_variance = np.var(Quantity_data)
Quantity_std_dev = np.std(Quantity_data)

print("Mean Quantity Level:", Quantity_mean)
print("Median Quantity Level:", Quantity_median)
print("Mode Quantity Level:", Quantity_mode)
print("Range of Quantity Levels:", Quantity_range)
print("Variance of Quantity Levels:", Quantity_variance)
print("Standard Deviation of Quanitys Levels:", Quantity_std_dev)

# 4

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data['Geography'], bins=20, kde=True)
plt.title('Histogram of profits accross Geographies')
plt.xlabel('Geography')
plt.ylabel('Profit')
plt.show()


fig = plt.figure(figsize = (10, 5))
plt.bar(data['Category'], data['Sales'], color ='maroon')
plt.xlabel("Category")
plt.ylabel("Sales")
plt.title("Sale per Category")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=data,x='Category', y='Quantity')
plt.title('Box plot of Category vs Sales')
plt.xlabel('Category')
plt.ylabel('Quantity')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Product Name', y='Sales')
plt.title('Scatter plot of BMI vs Length of Stay')
plt.xlabel('Product Name')
plt.ylabel('Sales')
plt.show()

plt.figure(figsize=(8, 8))
data['Category'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
plt.title('Percentage Per Category')
plt.ylabel('')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Geography')
plt.title('Count of Category')
plt.xlabel('List of categories')
plt.ylabel('Count')
plt.show()

# 5

correlation_matrix = data.corr()
print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

sns.pairplot(data)
plt.show()

# 6
Q7 = data.quantile(0.25)
Q8 = data.quantile(0.75)
IQR = Q8 - Q7
outliers = ((data < (Q7 - 1.5 * IQR)) | (data > (Q8 + 1.5 * IQR))).sum()
print(outliers)

plt.figure(figsize=(12, 8))
sns.boxplot(data=data)
plt.title('Boxplot of Numerical Variables')
plt.xticks(rotation=45)
plt.show()

# 7
data['year'] = pd.DatetimeIndex(data['Order Date']).year
plt.figure(figsize=(10, 6))

plt.plot(data['year'], data['Sales'], marker='o', color='blue', label='Sales')
plt.plot(data['year'], data['Profit'], marker='o', color='green', label='Profit')

plt.title('Sales and Profit Trends Over the Years')
plt.xlabel('year')
plt.ylabel('Sales')
plt.xticks(data['year'])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""#7.1.1
We can clearly see the profits and sale incresing over the years as we can see in the diagram 7.1 before taking a steep jump in year 2013 and 2014
Each year during the end of the year we always see a seasonal jump

# 7.1.2
We can clearly see that the category of phones and tables have taken the biggest jump in these years in terms of sales over the years after plotting a line plot
"""

# 7.2
# 7.2.1
top_customers_orders = data.groupby('Category')[data['Order ID']].count().nlargest(5)
top_customers_sales = data.groupby('Category')['Sales'].sum().nlargest(5)

print("Top 5 customers based on number of orders:")
print(top_customers_orders)
print("\nTop 5 customers based on total sales:")
print(top_customers_sales)

# 7.2.2
data['time_between_orders'] = data.groupby(data['Category'])['Order Date'].diff().dt.days
avg_time_between_orders = data.groupby('Category')['time_between_orders'].mean()

print("\nAverage time between orders for each customer:")
print(avg_time_between_orders)

"""# 7.2
#7.2.1
Majority of the top 5 customers in terms of sale were from the category sales
#7.2.2
Mostly the customers who had repeat orders had ordered very frequently from one date to another
"""

# 7.3
fig = plt.figure(figsize = (10, 5))
plt.bar(data['Order Date'], data['Sales'], color ='maroon')
plt.xlabel("Order Date")
plt.ylabel("Sales")
plt.title("Sale per order date")
plt.show()

"""#7.3.1
We can clearly see a jump in the amount of dates towards the end of the year so we can enhance the ccapacity of the supply chain accordingly for thse moths to meet enhanced demands as we can see in the plot in 7.3.1

# 7.3.2
 **Factors Contributing to Geographic Sales Distribution and Targeted Marketing:**

1. **Demographics:** Tailoring arketing strategies based on age, income levels, and preferences of different regions.
  
2. **Economic Conditions:** Adjusting marketing efforts according to local economic factors like income levels and employment rates.

3. **Cultural Preferences:** Adapting marketing messages and products to align with regional cultural norms and values.

5. **Infrastructure and Accessibiliy:** Considering accessibility and distribution channels to optimize sales efficiency.

6. **Seasonal Variations:** Developing seasonal marketing campaigns to capitalize on regional consumer trends and holidays.

#7.2.3

**Patterns and Predictors of High-Value Customers and Loyalty Strategies:**

1. **Purchase Behavior:** Analyzing frequency, recency, and monetary value of purchases to identify high-value segments.

3. **Customer Segmentation:** Tailoring marketing and loyalty programs based on demographic, psychographic, and behavioral characteristics.

4. **Personalized Marketing:** Delivering personalized communications and offers to enhance customer satisfaction and loyalty.

5. **Reward Programs:** Implementing loyalty points, discounts, and exclusive offers to incentivize repeat purchases.
"""