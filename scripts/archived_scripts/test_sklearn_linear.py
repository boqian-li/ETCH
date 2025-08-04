import numpy as np
from sklearn.linear_model import LinearRegression

# Generate sample data
N = 10  # Number of samples
X = np.random.rand(N, 4)  # Feature matrix, shape (N, 4)
gt_coef = np.array(range(4 * 3)).reshape(4, 3)
gt_intercept = np.array(range(3))

y = X @ gt_coef + gt_intercept

# Create linear regression model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Output regression coefficients and intercept
print("Coefficients:\n", model.coef_)
print("Intercept:\n", model.intercept_)

# Use model for prediction
y_pred = model.predict(X)
print("Predictions:\n", y_pred)
print("gt:\n", y)