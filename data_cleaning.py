
# ============================================================
# Task 2: Data Cleaning & Missing Value Handling
# Dataset: Titanic (titanic.csv)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# ------------------------------------------------------------
# STEP 1: Load the Titanic Dataset
# ------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv("titanic.csv")

print(f"Shape of dataset: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)


# ------------------------------------------------------------
# STEP 2: Identify Missing Values
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Identifying Missing Values")
print("=" * 60)

missing_counts = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_info = pd.DataFrame({
    "Missing Count": missing_counts,
    "Missing %": missing_percent
})

print(missing_info[missing_info["Missing Count"] > 0])


# ------------------------------------------------------------
# STEP 3: Visualize Missing Data using Seaborn Heatmap
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Visualizing Missing Data")
print("=" * 60)

plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap="viridis")
plt.title("Missing Value Heatmap (Yellow = Missing)", fontsize=14)
plt.tight_layout()
plt.savefig("missing_values_heatmap.png")
plt.show(block=False)
plt.pause(3)
plt.close()
print("Heatmap saved as 'missing_values_heatmap.png'")


# ------------------------------------------------------------
# STEP 4: Handle Missing Values
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Handling Missing Values")
print("=" * 60)

# --- 4a: Drop columns where missing data is too high (> 70%) ---
threshold = 70.0
cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()

if cols_to_drop:
    print(f"\nDropping columns with > {threshold}% missing data: {cols_to_drop}")
    df.drop(columns=cols_to_drop, inplace=True)
else:
    print(f"\nNo columns exceed the {threshold}% missing threshold. None dropped.")

# --- 4b: Separate numerical and categorical columns ---
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print(f"\nNumerical columns  : {numerical_cols}")
print(f"Categorical columns: {categorical_cols}")

# --- 4c: Impute numerical columns with Median ---
if numerical_cols:
    num_imputer = SimpleImputer(strategy="median")
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    print("\nNumerical missing values filled using MEDIAN.")

# --- 4d: Impute categorical columns with Most Frequent (Mode) ---
if categorical_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    print("Categorical missing values filled using MOST FREQUENT (Mode).")


# ------------------------------------------------------------
# STEP 5: Identify and Treat Outliers using IQR Method
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Outlier Detection & Treatment (IQR Capping)")
print("=" * 60)

# Target numerical columns likely to have outliers
outlier_cols = [col for col in ["Age", "Fare", "SibSp", "Parch"] if col in df.columns]

for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers before capping
    outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

    # Cap outliers (Winsorization)
    df[col] = np.clip(df[col], lower_bound, upper_bound)

    print(f"  '{col}': Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f} | "
          f"Bounds=[{lower_bound:.2f}, {upper_bound:.2f}] | "
          f"Outliers capped: {outliers_count}")

# Visualize after outlier treatment
fig, axes = plt.subplots(1, len(outlier_cols), figsize=(5 * len(outlier_cols), 4))
if len(outlier_cols) == 1:
    axes = [axes]
for ax, col in zip(axes, outlier_cols):
    ax.boxplot(df[col].dropna())
    ax.set_title(f"{col} (after capping)")
plt.suptitle("Boxplots After Outlier Treatment", fontsize=13)
plt.tight_layout()
plt.savefig("boxplots_after_outlier_treatment.png")
plt.show(block=False)
plt.pause(3)
plt.close()
print("\nBoxplot saved as 'boxplots_after_outlier_treatment.png'")


# ------------------------------------------------------------
# STEP 6: Verify – No Missing Values Remain
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Verifying Clean Dataset")
print("=" * 60)

remaining_missing = df.isnull().sum().sum()

if remaining_missing == 0:
    print("✅ Dataset is CLEAN — no missing values remain!")
else:
    print(f"⚠️  {remaining_missing} missing values still found. Please review.")

print(f"\nFinal dataset shape: {df.shape}")
print("\nMissing values per column:")
print(df.isnull().sum())


# ------------------------------------------------------------
# STEP 7: Save Cleaned Dataset
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: Saving Cleaned Dataset")
print("=" * 60)

df.to_csv("cleaned_titanic.csv", index=False)
print("✅ Cleaned dataset saved as 'cleaned_titanic.csv'")
print("=" * 60)
print("All steps completed successfully!")
print("=" * 60)
