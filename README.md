# Supermarket Sales Analysis

## Introduction

This project involves exploratory data analysis (EDA), visualizations, and machine learning model training/testing using a sales dataset.

## Table of Contents

- [About Dataset](#about-dataset)
- [Key Features](#key-features)
  - [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Machine Learning](#machine-learning)

## About Dataset

### Context
The dataset captures historical sales of a supermarket company in three different branches over a span of three months. Predictive data analytics methods can be applied to understand sales trends.

### Attribute Information
- **Invoice id:** Computer-generated sales slip invoice identification number
- **Branch:** Branch of supercenter (A, B, and C)
- ...

## Key Features

### Data Cleaning and Preprocessing

#### Overview

The dataset has been loaded and underwent necessary cleaning and preprocessing steps to ensure its suitability for analysis. This section provides an overview of the data cleaning process.

#### Steps Taken

1. **Loading the Dataset:**
   - The dataset, named `supermarket_sales.csv`, was loaded into a Pandas DataFrame.

2. **Handling Missing Values:**
   - No missing values were found in the dataset. All columns have complete data.

3. **Converting 'Date' to Datetime Format:**
   - The 'Date' column was converted to the datetime64 data type for better handling of date-related operations.

4. **Updated Data Information:**
   - After the data cleaning steps, the dataset information was displayed again to confirm the changes.

### Exploratory Data Analysis (EDA)

#### Customer Demographics
  - Explore the distribution of sales across different customer types (e.g., Member vs. Normal) and genders. Visualize the average sales or quantity purchased by each customer type.

![image](https://github.com/Lawrence-Mak/Sales-Analysis-Project/assets/83872954/371ed609-9497-436d-b3be-ec3f69b51417)

<details>
<summary style="color: blue;">Click to expand</summary>
  
```python
# Set the figure size
plt.figure(figsize=(16, 12))

# Plot 2: Distribution of Member and Non-Member
member_distribution = df['Customer type'].value_counts()
plt.subplot(3, 2, 1)
plt.pie(member_distribution, labels=member_distribution.index, autopct='%1.1f%%', colors=['sandybrown', 'darkseagreen'])
plt.title('Distribution of Customer Types (Member vs. Non-Member)')

# Plot 3: Total Sales by Customer Type
plt.subplot(3, 2, 2)
sns.barplot(x='Customer type', y='Total', data=df, estimator=sum, palette=['sandybrown', 'darkseagreen'])
plt.title('Total Sales by Customer Type')
plt.xlabel('Customer Type')
plt.ylabel('Total Sales')

# Plot 1: Popular Product Lines by Customer Type (taking up the entire bottom row)
plt.subplot(3, 2, (3, 4))
sns.countplot(x='Product line', hue='Customer type', data=df, palette={'Member': 'sandybrown', 'Normal': 'darkseagreen'})
plt.title('Popular Product Lines by Customer Type')
plt.xlabel('Product Line')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Customer Type')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
```

</details>

![image](https://github.com/Lawrence-Mak/Sales-Analysis-Project/assets/83872954/f9a5f537-6c9d-4f10-b057-7de7e960ef47)

<details>
<summary style="color: blue;">Code</summary>
  
```python
# Set the figure size
plt.figure(figsize=(16, 12))

# Plot 2: Distribution of Genders (pie chart)
gender_distribution = df['Gender'].value_counts()
plt.subplot(3, 2, 1)
plt.pie(gender_distribution, labels=gender_distribution.index, autopct='%1.1f%%', colors=['lightcoral', 'skyblue'])
plt.title('Distribution of Genders')

# Plot 4: Total Sales by Gender
plt.subplot(3, 2, 2)
sns.barplot(x='Gender', y='Total', data=df, estimator=sum, ci=None, palette=['lightcoral', 'skyblue'])
plt.title('Total Sales by Gender')
plt.xlabel('Gender')
plt.ylabel('Total Sales')

# Plot 6: Popular Product Lines by Gender (taking up the entire bottom row)
plt.subplot(3, 2, (3, 4))
sns.countplot(x='Product line', hue='Gender', data=df, palette={'Male': 'skyblue', 'Female': 'lightcoral'})
plt.title('Popular Product Lines by Gender')
plt.xlabel('Product Line')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Gender')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
```
</details>

#### Time-of-Day Analysis
- Analyze sales patterns based on the time of day.
  Explore whether there are specific times when sales peak or decline.

![image](https://github.com/Lawrence-Mak/Sales-Analysis-Project/assets/83872954/f4d43b1d-57f0-4f16-932a-a7aa1c6b4da7)

<details>
<summary style="color: blue;">Code</summary>
  
```python
# Convert 'Time' to datetime format
df['Time'] = pd.to_datetime(df['Time'])

# Set the figure size
plt.figure(figsize=(12, 6))

# Plot sales patterns based on the time of day
sns.lineplot(x=df['Time'].dt.hour, y='Total', data=df, estimator=sum, ci=None, color='skyblue')

# Set labels and title
plt.title('Sales Patterns Based on Time of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Total Sales')

# Customize x-axis ticks to start at 10 and end at 20
plt.xticks(range(10, 21))

# Show the plot
plt.show()
```
</details>



#### Branch Analysis
  - Investigate sales distribution across different branches. Analyze the performance of each branch in terms of total sales and customer satisfaction rating



#### Payment Method Analysis
- Examine the distribution of sales based on payment methods.
  Compare the average sales for different payment methods.

#### Unit Price and Quantity Analysis
 - Visualize the distribution of unit prices and quantities for products.
  Explore the relationship between unit price, quantity, and total sales.
  


#### Seasonal Trends
- Investigate sales trends across different seasons or months.
  Analyze whether certain product types are more popular during specific seasons.

#### Customer Ratings vs. Sales:
- Explore the relationship between customer ratings and total sales.
  Analyze whether higher-rated products lead to higher sales.

#### Promotions and Discounts:
- Investigate the impact of promotions or discounts on sales.
  Analyze whether there is a correlation between promotional periods and increased sales.

#### Customer Loyalty:
- Explore repeat purchases and customer loyalty.
  Analyze whether there are specific product types that attract repeat customers.

#### Price Elasticity:
- Analyze the relationship between changes in price and changes in quantity sold.
  Explore the price elasticity of demand for different product categories.


### Machine Learning

Apply predictive analytics methods to forecast sales trends.

### Visualization

Create visualizations to present insights in an understandable manner.
