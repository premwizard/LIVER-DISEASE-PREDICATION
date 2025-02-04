import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Filter the unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df1 = pd.read_csv('Liver_data.csv')
df = pd.DataFrame()
df['Age'] = df1['Age']
df['Gender'] = df1['Gender']
df['sugar'] = df1['sugar']
df['Direct_Bilirubin'] = df1['Direct_Bilirubin']
df['Alkaline_Phosphotase'] = df1['Alkaline_Phosphotase']
df['Alamine_Aminotransferase'] = df1['Alamine_Aminotransferase']
df['Aspartate_Aminotransferase'] = df1['Aspartate_Aminotransferase']
df['Total_Protiens'] = df1['Total_Protiens']
df['Albumin'] = df1['Albumin']
df['Albumin_and_Globulin_Ratio'] = df1['Albumin_and_Globulin_Ratio']
df['Result'] = df1['Result']

# Map the Result column
df['Result'] = df['Result'].map({2: 2, 1: 1})

# Check for NaN values in the DataFrame
print("Null values in the DataFrame:")
print(df.isnull().sum())

# Impute missing values using SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[df.columns[:-1]] = imputer.fit_transform(df[df.columns[:-1]])  # Impute all columns except 'Result'

# After handling NaN values, re-check for nulls
print("Null values after handling:")
print(df.isnull().sum())

# Proceed with the rest of the code
X = df.drop("Result", axis=1).values
y = df["Result"].values

# Check the shape of X and y
print("Shape of X:", X.shape)  # Should be (n_samples, 10)
print("Shape of y:", y.shape)  # Should be (n_samples,)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Check the shape of training and testing sets
print("Shape of X_train:", X_train.shape)  # Should be (n_samples_train, 10)
print("Shape of X_test:", X_test.shape)    # Should be (n_samples_test, 10)

error = []
# Will take some time
for i in range(550, 600):
    rfc = RandomForestClassifier(n_estimators=i)
    rfc.fit(X_train, y_train)
    pred_i = rfc.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# Fit the model with the optimal number of estimators
optimal_n_estimators = 571
rfc = RandomForestClassifier(n_estimators=optimal_n_estimators)
rfc.fit(X_train, y_train)

# Save the model
pickle.dump(rfc, open('model.pkl', 'wb'))

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Example of making predictions (ensure new data has the same number of features)
# new_data = np.array([[...], [...], ...])  # Replace with actual new data
# print("Shape of new_data:", new_data.shape)  # Should be (n_samples, 10)
# predictions = model.predict(new_data)