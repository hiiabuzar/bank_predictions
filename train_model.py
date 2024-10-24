import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
# Load the dataset
data = pd.read_csv("bank.csv")

# Encode categorical variables (excluding 'contact', 'duration', 'poutcome', 'campaign', and 'previous')
label_encoders = {}
for column in ['job', 'marital', 'education', 'default', 'housing', 'loan']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Convert target variable to binary (1 for 'yes', 0 for 'no')
data['y'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

data = data.drop(['contact', 'duration', 'poutcome', 'previous'], axis=1)

# Split the data into features and target
X = data.drop('y', axis=1)
y = data['y']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


# Save the model, scaler, and label encoders using pickle
with open('bank_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)
