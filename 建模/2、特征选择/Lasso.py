import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

##### 读取数据集
filename = ("训练数据集-有SMOTE.xlsx")
data = pd.read_excel(filename)

# 将数据集分为特征矩阵X和目标变量y
y = data["Target"]
X = data.loc[:, "SCC": "超氧歧化酶"]

# 初始化LassoCV模型
lasso_cv = LassoCV(cv=5)

# 拟合模型
lasso_cv.fit(X, y)

# 获取最佳的alpha值
best_alpha = lasso_cv.alpha_

# 初始化Lasso模型
lasso = Lasso(alpha=best_alpha)

# 拟合模型
lasso.fit(X, y)

# 打印特征系数
print("Feature weights:")
for feature, weight in zip(X.columns.values, lasso.coef_):
    print(f"{feature}: {weight}")

# 设置Seaborn风格和配色
sns.set(style="whitegrid", palette="dark")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.labelsize': 14,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.color': 'white'
})

# 绘制特征系数水平柱状图
fig, ax = plt.subplots(figsize=(8, 6))
coefficients = pd.Series(lasso.coef_, index=X.columns)
coefficients.plot(kind="barh", ax=ax)
ax.set_title("Feature Coefficients - LASSO Regression", fontsize=18)
ax.set_xlabel("Coefficient Value", fontsize=16)
ax.set_ylabel("Feature", fontsize=16)
ax.tick_params(axis="both", labelsize=14)
plt.show()

