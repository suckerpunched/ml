import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pickle

dataset = pd.read_csv('salary_data.csv')
x,y = dataset.iloc[:, :-1].values,dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

model = LinearRegression()
model.fit(x_train, y_train)

# ---

pickle.dump(model, open('model.pkl', 'wb'))

# ---
result = model.score(x_test, y_test)
print(result)

# visual the training set results
plt.figure(1)
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, model.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# visual the test set results
plt.figure(2)
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, model.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.show()