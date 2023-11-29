# Supermarket Sales Analysis

## Introduction

This repository contains code and documentation for analyzing historical sales data from a supermarket company. The dataset records sales from three different branches over a period of three months. The goal is to apply predictive data analytics methods to gain insights into sales trends.

## Table of Contents

- [About Dataset](#about-dataset)
- [Key Features](#key-features)
- [Usage](#usage)
- [Data Analysis](#data-analysis)

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

1. **Data Cleaning and Preprocessing:**
   - Clean and preprocess the dataset for analysis.
  
   - ## Data Cleaning and Preprocessing

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
   - Explore key statistics, trends, and patterns in the sales data.

3. **Predictive Analytics:**
   - Apply predictive analytics methods to forecast sales trends.

4. **Visualization:**
   - Create visualizations to present insights in an understandable manner.

