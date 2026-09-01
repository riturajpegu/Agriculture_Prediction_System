import pandas as pd

# Load the dataset
df = pd.read_csv("Custom_Crops_yield_Historical_Dataset.csv")
print(df.columns)

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Display information about the dataset
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

print("\nUnique Crops:")
print(df["Crop"].unique())

print("\nNumber of Crops:")
print(df["Crop"].nunique())

print("\nStates:")
print(df["State Name"].unique())

print("\nNumber of States:")
print(df["State Name"].nunique())

print("\nNumber of Districts:")
print(df["Dist Name"].nunique())

print("\nCrop Distribution:")
print(df["Crop"].value_counts())

print("\nAverage Climate Conditions by Crop:")
print(
    df.groupby("Crop")[
        ["Temperature_C", "Humidity_%", "Rainfall_mm",
         "pH", "Wind_Speed_m_s", "Solar_Radiation_MJ_m2_day"]
    ].mean().round(2)
)

print("\nCrop Distribution by State:")
print(pd.crosstab(df["State Name"], df["Crop"]))
