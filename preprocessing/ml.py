import numpy as np
import pandas as pd

dataset = pd.read_csv('data.csv')
x,y = dataset.iloc[:, :-1].values,dataset.iloc[:, -1].values

# handling missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])

x[:, 1:3] = imputer.transform(x[:, 1:3])

# encode categorical (features) data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# encode dependent variable (result)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

# ---
print("features\n", x)
print("result\n", y)