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

Explore the distribution of sales across different customer types (e.g., Member vs. Normal) and genders.
Visualize the average sales or quantity purchased by each customer type.

#### Branch Analysis

Investigate sales distribution across different branches.
Analyze the performance of each branch in terms of total sales and customer satisfaction ratings.

#### Payment Method Analysis

Examine the distribution of sales based on payment methods.
Compare the average sales for different payment methods.

#### Unit Price and Quantity Analysis

Visualize the distribution of unit prices and quantities for products.
Explore the relationship between unit price, quantity, and total sales.

#### Time-of-Day Analysis

Analyze sales patterns based on the time of day.
Explore whether there are specific times when sales peak or decline.

#### Seasonal Trends

Investigate sales trends across different seasons or months.
Analyze whether certain product types are more popular during specific seasons.

...

### Machine Learning

Apply predictive analytics methods to forecast sales trends.

### Visualization

Create visualizations to present insights in an understandable manner.
