import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and clean
df = pd.read_csv('healthcare_dataset.csv')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Feature engineering
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], dayfirst=True)
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], dayfirst=True)
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

df['Age Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior', 'Elder'])

# Assign risk score based on condition
risk_map = {'Cancer':3,'Diabetes':2,'Obesity':1}
df['Risk Score'] = df['Medical Condition'].map(risk_map).fillna(0)


# Drop columns
df = df.drop(['Name', 'Room Number', 'Doctor', 'Hospital', 'Date of Admission', 'Discharge Date'], axis=1)

# Encode and scale
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# One-hot encoding
df = pd.get_dummies(df, columns=[
    'Blood Type', 'Medical Condition', 'Insurance Provider',
    'Admission Type', 'Medication', 'Age Group'
])

# Scale numeric features
scaler = StandardScaler()
df[['Age', 'Length of Stay', 'Risk Score']] = scaler.fit_transform(df[['Age', 'Length of Stay', 'Risk Score']])

# Save processed data
print("Data preprocessing complete! Saved as processed_data.csv")