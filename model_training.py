import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib
from sklearn.impute import SimpleImputer # Import SimpleImputer

# Load processed data
df = pd.read_csv('processed_data.csv')
X = df.drop('Billing Amount', axis=1)
y = df['Billing Amount']

# Convert all columns to numeric, errors to NaN
for column in X.columns:
    try:
        X[column] = pd.to_numeric(X[column])
    except ValueError:
        X[column] = pd.to_numeric(X[column], errors='coerce')

# Impute NaN with column means using SimpleImputer
imputer = SimpleImputer(strategy='mean') # Create an imputer instance
X_imputed = imputer.fit_transform(X) # Fit and transform
# Get the columns that were kept after imputation
X_columns = X.columns[~pd.isnull(X).all()] # Only keep columns that are not all NaN
X = pd.DataFrame(X_imputed, columns=X_columns)  # Create DataFrame with correct columns

# Feature selection
selector = SelectKBest(score_func=f_regression, k=20)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_selected, columns=selected_features)

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42) # Fixed: Split X_temp and y_temp

# Build model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1)

# Save model and features
model.save('patient_cost_model.h5')
joblib.dump(selected_features, 'selected_features.pkl')
print("Model trained and saved!")