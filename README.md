# Supermarket Sales Analysis

## Introduction
This project involves exploratory data analysis (EDA), visualizations, and machine learning model training/testing using a sales dataset found here https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales.

## Table of Contents

- [Key Features](#key-features)
  - [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Customer Demographics](#Customer-Demographics)
    - [Time-of-Day Analysis](#Time-of-Day-Analysis)
    - [Branch Analysis](#Branch-Analysis)
    - [Payment Method Analysis](#Payment-Method-analysis)
  - [Machine Learning](#machine-learning)

## About Dataset

### Context
The dataset captures historical sales of a supermarket company in three different branches over a span of three months. Predictive data analytics methods can be applied to understand sales trends.

Attribute information:

Invoice id: Computer generated sales slip invoice identification number

Branch: Branch of supercenter (3 branches are available identified by A, B and C).

City: Location of supercenters

Customer type: Type of customers, recorded by Members for customers using member card and Normal for without member card.

Gender: Gender type of customer

Product line: General item categorization groups - Electronic accessories, Fashion accessories, Food and beverages, Health and beauty, Home and lifestyle, Sports and travel
Unit price: Price of each product in $
Quantity: Number of products purchased by customer
Tax: 5% tax fee for customer buying
Total: Total price including tax
Date: Date of purchase (Record available from January 2019 to March 2019)
Time: Purchase time (10 am to 9 pm)
Payment: Payment used by customer for purchase (3 methods are available – Cash, Credit card, and Ewallet)
COGS: Cost of goods sold
Gross margin percentage: Gross margin percentage
Gross income: Gross income
Rating: Customer stratification rating on their overall shopping experience (On a scale of 1 to 10)



## Key Features

### Data Cleaning and Preprocessing

#### Steps Taken
1. **Loading the Dataset:**
   - The dataset, named `supermarket_sales.csv`, was loaded into a Pandas DataFrame.
2. **Handling Missing Values:**
   - No missing values were found in the dataset. All columns have complete data.
3. **Converting 'Date' to Datetime Format:**
   - The 'Date' column was converted to the datetime64 data type for better handling of date-related operations.
4. **Updated Data Information:**
   - After the data cleaning steps, the dataset information is displayed again to confirm the changes.

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
![image](https://github.com/Lawrence-Mak/Sales-Analysis-Project/assets/83872954/c7657168-3451-4ffa-aba1-b25c0b88d34a)

<details>
<summary style="color: blue;">Code</summary>
  
```python
# Set the figure size
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Total Sales by Branch
sns.barplot(x='Branch', y='Total', data=df, estimator=sum, ci=None, palette='viridis', ax=axes[0])
axes[0].set_title('Total Sales by Branch')
axes[0].set_xlabel('Branch')
axes[0].set_ylabel('Total Sales')

# Plot 2: Average Rating by Branch
sns.barplot(x='Branch', y='Rating', data=df, estimator='mean', ci=None, palette='viridis', ax=axes[1])
axes[1].set_title('Average Rating by Branch')
axes[1].set_xlabel('Branch')
axes[1].set_ylabel('Average Rating')

# Adjust layout
plt.tight_layout()

# Show the combined plot
plt.show()
```
</details>


#### Payment Method Analysis
- Examine the distribution of sales based on payment methods.
  Compare the average sales for different payment methods.
  
![image](https://github.com/Lawrence-Mak/Sales-Analysis-Project/assets/83872954/d043a767-9256-46fb-b4c1-646d91c6c0f5)

<details>
<summary style="color: blue;">Code</summary>
  
```python
# Set the figure size
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Distribution of Sales by Payment Method
sns.countplot(x='Payment', data=df, palette='Set2', ax=axes[0])
axes[0].set_title('Distribution of Sales by Payment Method')
axes[0].set_xlabel('Payment Method')
axes[0].set_ylabel('Count')

# Plot 2: Average Sales by Payment Method
sns.barplot(x='Payment', y='Total', data=df, estimator='mean', ci=None, palette='Set2', ax=axes[1])
axes[1].set_title('Average Sales by Payment Method')
axes[1].set_xlabel('Payment Method')
axes[1].set_ylabel('Average Sales')

# Rotate x-axis labels for better visibility
for ax in axes:
    ax.tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()

# Show the combined plot
plt.show()
```
</details>


### Machine Learning
Apply predictive analytics methods to forecast sales trends.

We will creating a machine learning model to predict total sales. 

Step 1: Feature selection. 
Identify the most significant factors that impact total sales.
Use techniques like correlation analysis, feature importance from tree-based models, or statistical tests.

We will be using Correlation Analysis to determine correlated features

```python
# Encode categorical variables 
df_encoded = pd.get_dummies(df, columns=['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment'])

# Calculate correlation matrix
correlation_matrix = df_encoded.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix[['Total']], annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()
```
![image](https://github.com/Lawrence-Mak/Sales-Analysis-Project/assets/83872954/bd589800-94c7-439e-bb59-9f676de47475)

We can see that tax, cogs (cost of goods sold) and gross income have a perfect correlation of 1, it implies that they are linearly dependent. In a predictive model, using all three of them might introduce multicollinearity issues, where one variable can be predicted perfectly from the others. 

Therefore, its best to choose only one of these variables to represent the information they collevtively carry. 
```python
# Check pairwise correlation
correlation_tax_cogs = df[['Tax 5%', 'cogs']].corr().iloc[0, 1]
correlation_tax_gross = df[['Tax 5%', 'gross income']].corr().iloc[0, 1]
correlation_cogs_gross = df[['cogs', 'gross income']].corr().iloc[0, 1]

# Decide which variable to keep based on correlations
if correlation_tax_cogs >= correlation_tax_gross and correlation_tax_cogs >= correlation_cogs_gross:
    # Keep 'Tax'
    df.drop(['cogs', 'gross income'], axis=1, inplace=True)
    kept_variable = 'Tax'
elif correlation_tax_gross >= correlation_tax_cogs and correlation_tax_gross >= correlation_cogs_gross:
    # Keep 'gross income'
    df.drop(['Tax 5%', 'cogs'], axis=1, inplace=True)
    kept_variable = 'Gross Income'
else:
    # Keep 'cogs'
    df.drop(['Tax 5%', 'gross income'], axis=1, inplace=True)
    kept_variable = 'COGS'

print(f"The code ends up keeping: {kept_variable}")
```

This code checks the pairwise correlations between Tax, COGS, and Gross Income and decides which one to keep based on the highest correlation. In this case, we keep Gross Income as our predictive variable. 

We will therefore use Unit Price, Quantity, and Gross Income as predictors for our regression analysis. 


Train-Test Split:
Split dataset into training and testing sets. The training set is used to train the model, and the testing set is used to evaluate its performance.

```python
X = df[['gross income', 'Unit price', 'Quantity']]  # Features
y = df['Total']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting sets
print("Training set - Features:", X_train.shape, "Labels:", y_train.shape)
print("Testing set - Features:", X_test.shape, "Labels:", y_test.shape)
```

Model Selection and Training
Choose a regression model suitable for predicting total sales. We will train using a linear regression model
```python
# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Optionally, you can visualize the predicted vs. actual values
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Total Sales")
plt.ylabel("Predicted Total Sales")
plt.title("Actual vs. Predicted Total Sales")
plt.show()
```
Mean Squared Error: 4.378067583449884e-27
R-squared: 1.0
![image](https://github.com/Lawrence-Mak/Sales-Analysis-Project/assets/83872954/b63403cd-e281-4bf1-bcf4-fd924e255ab8)


Model Evaluation:
Mean Squared Error: 4.378067583449884e-27 This is an extremely low MSE, which means the model's predictions are very close to the actual values. In fact, the predicted values are almost identical to the true values.
R-squared: 1.0 Ranges from 0 to 1, where 1 indicates a perfect fit. An R-squared of 1.0 means that the model explains all the variability of the response data around its mean. In other words, the model perfectly predicts the target variable.

These results suggest that the model is performing exceptionally well on the testing dataset. However, it's essential to keep in mind that such perfect performance might be a sign of overfitting. Its almost impossible to achieve a perfect fit, therefore to the best of my knowledge this dataset has been created, instead of scraped from real world scenarios.

In sumamry, we have created a perfect model, however there are many implications that this would never happen in the real world. Real-world data often contains noise, outliers, and missing values. In the first place, this dataset did not include any outliers, missing data, or noise. It is very likely that this dataset was generated synthetically for educational or illustrative purposes, it might not accurately represent a real-world scenario.












