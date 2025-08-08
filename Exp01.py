import seaborn as sns
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split



df = pd.read_csv('Titanic.csv')


print("Initial Data:")
print(df.head())


print("\nData Info:")
df.info()

print("\nSummary Statistics for Numerical Columns:")
print(df.describe())

print("\nMissing Values Per Column:")
print(df.isnull().sum())

#  Handle Missing Data


numerical_cols = ['Age', 'Fare']
numerical_imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

categorical_cols = ['Embarked']
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

df = df.dropna(subset=['Survived'])

print("\nMissing Values After Imputation:")
print(df.isnull().sum())


label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])

# One-Hot Encoding for nominal categorical columns like 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Display the data after encoding
print("\nData After Encoding Categorical Variables:")
print(df.head())

#  Apply Normalization or Standardization

# We will standardize numerical features like 'Age' and 'Fare'
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print("\nData After Standardization (Age and Fare):")
print(df[['Age', 'Fare']].head())

#  Split Dataset into Train/Test

# Define features (X) and target variable (y)
X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of training and testing sets
print("\nTraining and Testing Set Shapes:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

#  Final Output

print("\nPreprocessed Data:")
print(df.head())
