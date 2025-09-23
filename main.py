import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])  # soatlar
y = np.array([2, 4, 6, 8, 10])  

model = LinearRegression()
model.fit(X, y)

print("w (koeff):", model.coef_)
print("b (intercept):", model.intercept_)

# Bashorat
print("3 soat o'qisa natija:", model.predict([[25]])) 



# Always show details
import matplotlib.pyplot as plt
import numpy as np

# Data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Regression coefficients
b1 = 0.6
b0 = 2.2

# Regression line
x_line = np.linspace(0, 6, 100)
y_line = b0 + b1 * x_line

# Plot
plt.scatter(x, y, color="blue", label="Haqiqiy nuqtalar")
plt.plot(x_line, y_line, color="red", label="Regressiya chizig'i")
plt.xlabel("O‘qish soati (X)")
plt.ylabel("Imtihon bali (Y)")
plt.title("Least Squares Regression misoli")
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Regression coefficients (from previous calculation)
b1 = 0.6
b0 = 2.2

# Predicted values and residuals
y_pred = b0 + b1 * x
residuals = y - y_pred
sse = np.sum(residuals**2)

# Regression line for plotting
x_line = np.linspace(0, 6, 200)
y_line = b0 + b1 * x_line

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Haqiqiy nuqtalar")
plt.plot(x_line, y_line, label="Regressiya chizig'i")
# Draw residuals as vertical lines
for xi, yi, ypi in zip(x, y, y_pred):
    plt.vlines(xi, ypi, yi)
    
plt.xlabel("O‘qish soati (X)")
plt.ylabel("Imtihon bali (Y)")
plt.title("Nuqtalar, regressiya chizig'i va residuallar (qoldiqlar)")
plt.legend()
plt.grid(True)
plt.xlim(0, 6)
plt.ylim(1.5, 6)
plt.show()

# Print residuals and SSE
print("x:", x.tolist())
print("y (haqiqiy):", y.tolist())
print("y_pred (bashorat):", np.round(y_pred, 3).tolist())
print("Residuallar (y - y_pred):", np.round(residuals, 3).tolist())
print("SSE (sum of squared errors):", round(float(sse), 3))
