import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("C:/Users/HP/Downloads/student_stress_model_20 - Sheet1.csv")
print(df.head()) #first 5 rows

features=['anxiety_level','self_esteem','sleep_quality','study_load','peer_pressure','social_support','future_career_concerns']
df['stress_level'] = df['stress_level'].map({'Low': 0, 'Moderate': 1, 'High': 2})
x=df[features]
y=df['stress_level']

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)

model=RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("\n Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

def predict_stress(user_input):
    result = model.predict(input_df)
    mapping = {0:'Low', 1:'Moderate', 2:'High'}  
    return mapping[result[0]]

def compare_to_average(user_input):
    print("\nComparison with dataset averages:")
    for i, feature in enumerate(features):
        avg = x[feature].mean()  # calculate average of each column
        print(f"{feature}: Your value = {user_input[i]}, Average = {avg:.2f}")
        
user_input = []
for feature in features:
    val = float(input(f"Enter your {feature.replace('_', ' ')} (1-10): "))
    user_input.append(val)
input_df = pd.DataFrame([user_input], columns=x_train.columns)


stress = predict_stress(user_input)
print(f"\nYour predicted stress level is: {stress}")
compare_to_average(user_input)











        





