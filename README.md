# Supermarket Sales Analysis

## Introduction

This project involves exploratory data analysis (EDA), visualizations, and machine learning model training/testing using a sales dataset.

## Table of Contents

- [About Dataset](#about-dataset)
- [Key Features](#key-features)
   - [EDA]

## About Dataset

### Context
The dataset captures historical sales of a supermarket company in three different branches over a span of three months. Predictive data analytics methods can be applied to understand sales trends.

### Attribute Information
- **Invoice id:** Computer-generated sales slip invoice identification number
- **Branch:** Branch of supercenter (A, B, and C)
- **City:** Location of supercenters
- **Customer type:** Type of customers (Member or Normal)
- **Gender:** Gender type of customer
- **Product line:** General item categorization groups
- **Unit price:** Price of each product in $
- **Quantity:** Number of products purchased by customer
- **Tax:** 5% tax fee for customer buying
- **Total:** Total price including tax
- **Date:** Date of purchase (January 2019 to March 2019)
- **Time:** Purchase time (10 am to 9 pm)
- **Payment:** Payment used by customer for purchase (Cash, Credit card, Ewallet)
- **COGS:** Cost of goods sold
- **Gross margin percentage:** Gross margin percentage
- **Gross income:** Gross income
- **Rating:** Customer stratification rating on their overall shopping experience (1 to 10)

### Purpose
This dataset is suitable for predictive data analytics, offering opportunities to analyze and predict supermarket sales trends.

### Acknowledgements 
The data used for this project can be found at https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales/ Thank you for the author Aung Pyae for collecting this data.

## Key Features

- [Data Cleaning and Preprocessing](#Data-Cleaning-and-Preprocessing)
- [Exploratory Data Analysis](#Exploratory-data-Analysis-(EDA))
- [Machine Learning]

### Data Cleaning and Preprocessing

### Overview

The dataset has been loaded and underwent necessary cleaning and preprocessing steps to ensure its suitability for analysis. This section provides an overview of the data cleaning process.

### Steps Taken

1. **Loading the Dataset:**
   - The dataset, named `supermarket_sales.csv`, was loaded into a Pandas DataFrame.

2. **Handling Missing Values:**
   - No missing values were found in the dataset. All columns have complete data.

3. **Converting 'Date' to Datetime Format:**
   - The 'Date' column was converted to the datetime64 data type for better handling of date-related operations.

4. **Updated Data Information:**
   - After the data cleaning steps, the dataset information was displayed again to confirm the changes.

### Updated Dataset Information

The dataset now includes the following data types:

- Object: Invoice ID, Branch, City, Customer type, Gender, Product line, Time, Payment
- Float64: Unit price, Tax 5%, Total, cogs, gross margin percentage, gross income, Rating
- Int64: Quantity
- Datetime64: Date

The cleaned dataset has been saved as `cleaned_supermarket_sales.csv` for further analysis.

2. **Exploratory Data Analysis (EDA):**

Customer Demographics:

Explore the distribution of sales across different customer types (e.g., Member vs. Normal) and genders.
Visualize the average sales or quantity purchased by each customer type.

Branch Analysis:

Investigate sales distribution across different branches.
Analyze the performance of each branch in terms of total sales and customer satisfaction ratings.

Payment Method Analysis:

Examine the distribution of sales based on payment methods.
Compare the average sales for different payment methods.

Unit Price and Quantity Analysis:

Visualize the distribution of unit prices and quantities for products.
Explore the relationship between unit price, quantity, and total sales.

Time-of-Day Analysis:

Analyze sales patterns based on the time of day.
Explore whether there are specific times when sales peak or decline.

Seasonal Trends:

Investigate sales trends across different seasons or months.
Analyze whether certain product types are more popular during specific seasons.

Customer Ratings vs. Sales:

Explore the relationship between customer ratings and total sales.
Analyze whether higher-rated products lead to higher sales.

Promotions and Discounts:

Investigate the impact of promotions or discounts on sales.
Analyze whether there is a correlation between promotional periods and increased sales.

Customer Loyalty:

Explore repeat purchases and customer loyalty.
Analyze whether there are specific product types that attract repeat customers.

Price Elasticity:

Analyze the relationship between changes in price and changes in quantity sold.
Explore the price elasticity of demand for different product categories.


3. **Predictive Analytics:**
   - Apply predictive analytics methods to forecast sales trends.

4. **Visualization:**
   - Create visualizations to present insights in an understandable manner.

