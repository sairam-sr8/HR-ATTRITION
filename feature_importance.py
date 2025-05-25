import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('HR-Employee-Attrition.csv')
df = df.drop(columns=['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'])
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
X = df.drop('Attrition', axis=1)
y = df['Attrition']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
importances = model.feature_importances_
features = X.columns
sorted_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_features[:6]:
    print(f'{feature}: {importance:.4f}')
