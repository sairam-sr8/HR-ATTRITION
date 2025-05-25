#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


dataset = pd.read_csv("HR-Employee-Attrition.csv")


# In[8]:


dataset.head()


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[11]:


# Load the dataset
df = pd.read_csv("HR-Employee-Attrition.csv")


# In[12]:


# Drop irrelevant or constant columns
df = df.drop(columns=["EmployeeNumber", "Over18", "StandardHours", "EmployeeCount"])


# In[13]:


# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])


# In[14]:


# Define features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]


# In[15]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[17]:


# Predictions
y_pred = model.predict(X_test)


# In[18]:


# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[26]:


dataset.isnull().sum()


# In[ ]:




