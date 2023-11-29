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
```# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

# Plot 1: Distribution of Customer Types
sns.countplot(x='Customer type', data=df, palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Customer Types')
axes[0, 0].set_xlabel('Customer Type')
axes[0, 0].set_ylabel('Count')

# Plot 2: Distribution of Genders
gender_distribution = df['Gender'].value_counts()
axes[0, 1].pie(gender_distribution, labels=gender_distribution.index, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
axes[0, 1].set_title('Distribution of Genders')

# Plot 3: Total Sales by Customer Type
sns.barplot(x='Customer type', y='Total', data=df, estimator=sum, ci=None, palette='viridis', ax=axes[1, 0])
axes[1, 0].set_title('Total Sales by Customer Type')
axes[1, 0].set_xlabel('Customer Type')
axes[1, 0].set_ylabel('Total Sales')

# Plot 4: Total Sales by Gender
sns.barplot(x='Gender', y='Total', data=df, estimator=sum, ci=None, palette='viridis', ax=axes[1, 1])
axes[1, 1].set_title('Total Sales by Gender')
axes[1, 1].set_xlabel('Gender')
axes[1, 1].set_ylabel('Total Sales')

# Adjust layout
plt.tight_layout()
plt.show()
```

#### Branch Analysis
  - Investigate sales distribution across different branches. Analyze the performance of each branch in terms of total sales and customer satisfaction ratings.

#### Payment Method Analysis
- Examine the distribution of sales based on payment methods.
  Compare the average sales for different payment methods.

#### Unit Price and Quantity Analysis
 - Visualize the distribution of unit prices and quantities for products.
  Explore the relationship between unit price, quantity, and total sales.
  
#### Time-of-Day Analysis
- Analyze sales patterns based on the time of day.
  Explore whether there are specific times when sales peak or decline.

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
