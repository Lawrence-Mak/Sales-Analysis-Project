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
- 

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

A preview of the dataset:

| Invoice ID   | Branch | City      | Customer type | Gender | Product line          | Unit price | Quantity | Tax 5% | Total   | Date      | Time  | Payment       | cogs  | gross margin percentage | gross income | Rating |
|--------------|--------|-----------|---------------|--------|-----------------------|------------|----------|--------|---------|-----------|-------|---------------|-------|--------------------------|---------------|--------|
| 750-67-8428  | A      | Yangon    | Member        | Female | Health and beauty     | 74.69      | 7        | 26.1415| 548.9715| 1/5/2019  | 13:08 | Ewallet        | 522.83| 4.761904762              | 26.1415       | 9.1    |
| 226-31-3081  | C      | Naypyitaw | Normal        | Female | Electronic accessories| 15.28      | 5        | 3.82   | 80.22  | 3/8/2019  | 10:29 | Cash          | 76.4  | 4.761904762              | 3.82          | 9.6    |
| 631-41-3108  | A      | Yangon    | Normal        | Male   | Home and lifestyle    | 46.33      | 7        | 16.2155| 340.5255| 3/3/2019  | 13:23 | Credit card   | 324.31| 4.761904762              | 16.2155       | 7.4    |
| 123-19-1176  | A      | Yangon    | Member        | Male   | Health and beauty     | 58.22      | 8        | 23.288 | 489.048 | 1/27/2019 | 20:33 | Ewallet        | 465.76| 4.761904762              | 23.288        | 8.4    |
| 373-73-7910  | A      | Yangon    | Normal        | Male   | Sports and travel     | 86.31      | 7        | 30.2085| 634.3785| 2/8/2019  | 10:37 | Ewallet        | 604.17| 4.761904762              | 30.2085       | 5.3    |
| 699-14-3026  | C      | Naypyitaw | Normal        | Male   | Electronic accessories| 85.39      | 7        | 29.8865| 627.6165| 3/25/2019 | 18:30 | Ewallet        | 597.73| 4.761904762              | 29.8865       | 4.1    |
| 355-53-5943  | A      | Yangon    | Member        | Female | Electronic accessories| 68.84      | 6        | 20.652 | 433.692 | 2/25/2019 | 14:36 | Ewallet        | 413.04| 4.761904762              | 20.652        | 5.8    |
| 315-22-5665  | C      | Naypyitaw | Normal        | Female | Home and lifestyle    | 73.56      | 10       | 36.78  | 772.38  | 2/24/2019 | 11:38 | Ewallet        | 735.6 | 4.761904762              | 36.78         | 8      |
| 665-32-9167  | A      | Yangon    | Member        | Female | Health and beauty     | 36.26      | 2        | 3.626  | 76.146 | 1/10/2019 | 17:15 | Credit card   | 72.52 | 4.761904762              | 3.626         | 7.2    |
| 692-92-5582  | B      | Mandalay  | Member        | Female | Food and beverages    | 54.84      | 3        | 8.226  | 172.746| 2/20/2019 | 13:27 | Credit card   | 164.52| 4.761904762              | 8.226         | 5.9    |



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
  




### Machine Learning

Apply predictive analytics methods to forecast sales trends.

### Visualization

Create visualizations to present insights in an understandable manner.
