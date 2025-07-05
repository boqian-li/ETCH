import numpy as np
from sklearn.linear_model import LinearRegression

# 生成示例数据
N = 10  # 样本数量
X = np.random.rand(N, 4)  # 特征矩阵，形状为 (N, 4)
gt_coef = np.array(range(4 * 3)).reshape(4, 3)
gt_intercept = np.array(range(3))

y = X @ gt_coef + gt_intercept

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 输出回归系数和截距
print("Coefficients:\n", model.coef_)
print("Intercept:\n", model.intercept_)

# 使用模型进行预测
y_pred = model.predict(X)
print("Predictions:\n", y_pred)
print("gt:\n", y)