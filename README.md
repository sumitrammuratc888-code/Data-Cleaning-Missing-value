# Data-Cleaning-Missing-value
# Edutech Solution AI & ML Internship - Task 2

## Data Cleaning & Missing Value Handling

*Objective:* To perform data cleaning, handle missing values, and treat outliers on the Titanic dataset to prepare it for a Machine Learning pipeline.

*Dataset Used:* Titanic Dataset

*Tools & Libraries Used:*
* Python
* Pandas & NumPy (Data manipulation)
* Matplotlib & Seaborn (Data visualization)
* Scikit-Learn (For missing value imputation)

*Steps Performed:*
1. *Identification:* Used df.isnull().sum() to identify columns with missing data.
2. *Visualization:* Plotted a Seaborn heatmap to visualize the distribution of missing values before cleaning.
3. *Handling Missing Values:*
   * *Deletion:* Dropped the Cabin column automatically as it had a very high percentage of missing data (~77%).
   * *Imputation:* Used Scikit-Learn's SimpleImputer to fill missing values in the Age (numerical) and Embarked (categorical) columns.
4. *Outlier Treatment:* Identified and capped outliers in numerical columns using the IQR (Interquartile Range) method to reduce data noise.
5. *Verification:* Confirmed that the dataset has zero missing values remaining.
6. *Final Output:* Exported the fully cleaned dataset as cleaned_titanic.csv.

*Repository Contents:*
* data_cleaning.py: The main Python script containing the code.
* titanic.csv: The original, raw dataset.
* cleaned_titanic.csv: The final, processed dataset ready for ML models.
* missing_values_heatmap.png: Screenshot demonstrating the missing data visualization.
* boxplots_after_outlier_treatment.png: Screenshot showing the data after handling outliers.
