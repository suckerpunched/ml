import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import pickle

dataset = pd.read_csv('./data/Position_Salaries.csv')
x,y = dataset.iloc[:, 1:-1].values,dataset.iloc[:, -1].values

reg_model = LinearRegression()
reg_model.fit(x, y)

poly_feat = PolynomialFeatures(degree= 4 )
x_poly = poly_feat.fit_transform(x)

poly_reg_model = LinearRegression()
poly_reg_model.fit(x_poly, y)

# ---

print(reg_model.predict([[6.5]]))
print(poly_reg_model.predict(poly_feat.fit_transform([[6.5]])))

# Visualising the Linear Regression results
plt.figure(1)
plt.scatter(x, y, color = 'red')
plt.plot(x, reg_model.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

# Visualising the Polynomial Regression results
plt.figure(2)
plt.scatter(x, y, color = 'red')
plt.plot(x, poly_reg_model.predict(poly_feat.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

# ---

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
plt.figure(3)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, poly_reg_model.predict(poly_feat.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression) (Smooth)')
plt.xlabel('Position level')
plt.ylabel('Salary')

plt.show()

# ---

