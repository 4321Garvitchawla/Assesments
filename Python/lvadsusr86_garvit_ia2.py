# -*- coding: utf-8 -*-
"""LVADSUSR86-garvit-IA2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Nmivok1ZFoTsG5rnJwoGn0VpChBUXhPL
"""

# 1
import numpy as np

def rgb_to_gray_luminosity(rgb_image):
    gray_image = np.dot(rgb_image[...,:3], [0.2989, 0.5870, 0.1140])

    return gray_image
rgb_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                      [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
                      [[127, 127, 127], [200, 200, 200], [50, 50, 50]]])

gray_image = rgb_to_gray_luminosity(rgb_image)

print("Original RGB Image:")
print(rgb_image)
print("\nGrayscale Image using Luminosity Method:")
print(gray_image)

# 2

import numpy as np

def normalize_data(data):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    normalized_data = (data - means) / stds
    return normalized_data
data = { 'Age': [25, 30, 35, 40, 45, 50, 55],
'Height':[172,179,178,166,190,191,146]}
normalize_data(data)

# 3
import numpy as np
sensor1_data = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])

sensor2_data = np.array([[10, 11, 12],
                         [13, 14, 15],
                         [16, 17, 18]])

sensor3_data = np.array([[19, 20, 21],
                         [22, 23, 24],
                         [25, 26, 27]])

sensor_data = np.stack([sensor1_data, sensor2_data, sensor3_data])
flattened_data = sensor_data.reshape(sensor_data.shape[0], -1)
reshaped_data = flattened_data.reshape(sensor_data.shape[0], -1)

print("Original 3D Array (Sensor Data):")
print(sensor_data)
print("\nFlattened Data:")
print(flattened_data)
print("\nReshaped Data:")
print(reshaped_data)

# 4

import numpy as np
first_game_index = 0
last_game_index = -1

first_game_scores = data[:, first_game_index]
last_game_scores = data[:, last_game_index]

improvement = last_game_scores - first_game_scores

data = { 'first_game_scores': [25, 30, 35, 40, 45, 50, 55],
'last_game_scores':[172,179,178,166,190,191,146]}

# 5
import numpy as np

scores = np.array([
    [90, 85, 92, 88, -1],
    [75, 80, -1, 85, 90],
    [88, 92, 90, -1, -1],
    [82, 78, 80, 85, 88]
])
def average_last_three_subjects(row):
    valid_scores = row[row != -1]
    return np.mean(valid_scores[-3:])
average_scores = np.apply_along_axis(average_last_three_subjects, 1, scores)

print("Average scores in the last three subjects for each student:")
print(average_scores)

# 6

def apply_adjustment_factors(city_temperatures, adjustment_factors):
    assert adjustment_factors.shape[0] == city_temperatures.shape[1], "Adjustment factors dimensions do not match city temperatures"
    adjusted_temperatures = city_temperatures * adjustment_factors.T
    return adjusted_temperatures

city_temperatures = np.random.rand(5, 12) * 50
adjustment_factors = np.random.rand(12) * 0.2 + 0.9

adjusted_temperatures = apply_adjustment_factors(city_temperatures, adjustment_factors)
print(adjusted_temperatures)

# 7
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'], 'Age': [25, 30, 35, 40, 45, 50, 55],
'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Miami', 'Boston'],
'Department': ['HR', 'IT', 'Finance', 'Marketing', 'Sales', 'IT', 'HR']}

employee_data = pd.DataFrame(data)
selected_rows = employee_data[(employee_data['Age'] > 45) & (employee_data['Department'] != 'HR')]
selected_columns = selected_rows[['Name','City']]
print(selected_columns)

# 8

data = [
    {"Product": "Apples", "Category": "Fruit", "Price": 1.20, "Promotion": True},
    {"Product": "Bananas", "Category": "Fruit", "Price": 0.50, "Promotion": False},
    {"Product": "Cherries", "Category": "Fruit", "Price": 3.00, "Promotion": True},
    {"Product": "Dates", "Category": "Fruit", "Price": 2.50, "Promotion": True},
    {"Product": "Elderberries", "Category": "Fruit", "Price": 4.00, "Promotion": False},
    {"Product": "Flour", "Category": "Bakery", "Price": 1.50, "Promotion": True},
    {"Product": "Grapes", "Category": "Fruit", "Price": 2.00, "Promotion": False}
]

fruit_prices = [product["Price"] for product in data if product["Category"] == "Fruit"]
average_fruit_price = sum(fruit_prices) / len(fruit_prices)

potential_candidates = []
for product in data:
    if product["Category"] == "Fruit" and product["Price"] > average_fruit_price and not product["Promotion"]:
        potential_candidates.append(product["Product"])

print("Potential candidates for future promotions:")
for candidate in potential_candidates:
    print(candidate)

# 9
import pandas as pd

employee_data = {
    'Employee': ['Alice', 'Bob', 'Charlie', 'David'],
    'Department': ['HR', 'IT', 'Finance', 'IT'],
    'Manager': ['John', 'Rachel', 'Emily', 'Rachel']
}

project_data = {
    'Employee': ['Alice', 'Charlie', 'Eve'],
    'Project': ['P1', 'P3', 'P2']
}

employee_df = pd.DataFrame(employee_data)
project_df = pd.DataFrame(project_data)

merged_df = pd.merge(project_df, employee_df, on='Employee', how='left')

merged_df['Department'].fillna('Unassigned', inplace=True)
merged_df['Manager'].fillna('Unassigned', inplace=True)

department_overview = merged_df.groupby('Department').agg({'Employee': list, 'Project': list, 'Manager': 'first'}).reset_index()

print("Departmental Overview:")
print(department_overview)

# 10
data = {
    'Department': ['Electronics', 'Electronics', 'Clothing', 'Clothing', 'Home Goods'],
    'Salesperson': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Sales': [70000, 50000, 30000, 40000, 60000]
}
df = pd.DataFrame(data)
total_sales_per_dept = df.groupby('Department')['Sales'].sum()
salespeople_per_dept = df.groupby('Department')['Salesperson'].count()
average_sales_per_salesperson = total_sales_per_dept / salespeople_per_dept
ranked_departments = average_sales_per_salesperson.sort_values(ascending=False)

print("Average Sales per Salesperson in Each Department:")
print(average_sales_per_salesperson)
print("\nRanking of Departments based on Average Sales per Salesperson:")
print(ranked_departments)