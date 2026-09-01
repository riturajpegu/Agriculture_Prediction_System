import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# Read dataset
data = pd.read_csv(r"C:\Users\ritur\Downloads\Custom_Crops_yield_Historical_Dataset.csv")

print(data.head())
print(data.shape)


# X and y
X = data[[
    "Temperature_C",
    "Humidity_%",
    "pH",
    "Rainfall_mm",
    "Wind_Speed_m_s",
    "Solar_Radiation_MJ_m2_day"
]]

y = data["Crop"]


# Convert crop names into numbers
encoder = LabelEncoder()

y = encoder.fit_transform(y)


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# KNN
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("KNN Accuracy:", accuracy)


# SVM
model = SVC()

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("SVM Accuracy:", accuracy)


# Naive Bayes
model = GaussianNB()

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)


print("Naive Bayes Accuracy:", accuracy)


# Save KNN model
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

pickle.dump(model, open("crop_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("Model Saved")
