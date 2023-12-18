import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt #可视化模块
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #z-score归一化
from sklearn.preprocessing import MinMaxScaler   #Min-Max归一化
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc

# 读取Excel数据
filename = ("训练数据集-有SMOTE.xlsx")
df = pd.read_excel(filename)

# 将特征赋值给自变量x，将标签赋值给因变量
y = df["Target"]
x = df.loc[:, "SCC": "超氧歧化酶"]

#将上一步得到的数据划分为训练集和测试集两部分（参数test_size为浮点数时，表示测试集的占比：如果是整数，表示测试集的样本数）
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print("训练集形态;", X_train.shape)
print("测试集数据形态：", X_test.shape)

#数据标准化
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.fit_transform(X_test)

#数据归一化
#MinMaxScaler = MinMaxScaler(feature_range=[0,1])  #设定数据转换的范围是[0,1]
#X_train      = MinMaxScaler.fit_transform(X_train)
#X_test       = MinMaxScaler.fit_transform(X_test)

#1.首先，不管任何参数，都选择默认，我们先拟合下数据看看：
RF_model = RandomForestClassifier(oob_score=True, random_state=16)
RF_model.fit(X_train, y_train.astype('int'))

#----------------------------------------------------------------#
#                              k折交叉验证                         #
#----------------------------------------------------------------#
#训练集和测试集的划分在一定程度上影响该模型性能的，
#仅仅是划分一次训练集和测试集的模型是不具有强说服力的，
#需要根据k折交叉验证的结果对模型进行选择或优化。
#导入K折交叉验证模块
#使用K折交叉验证模块，把数据集划分成k份子集，每次将K-1份子集作为训练集，1份数据作为测试集，
#这样我们就得到k组训练集和测试集，进行k次训练和测试，最终返回k组测试的均值。

import sklearn.model_selection as ms

#k折交叉验证
accuracy = ms.cross_val_score(RF_model, x, y, cv=10, scoring='accuracy')
print('CV Accuracy:', accuracy.mean(), accuracy.std())
pw = ms.cross_val_score(RF_model, x, y, cv=10, scoring='precision_weighted')
print('CV Specificity:', pw.mean(), pw.std())
rw = ms.cross_val_score(RF_model, x, y, cv=10, scoring='recall_weighted')
print('CV Sensitivity:', rw.mean(), rw.std())
fw = ms.cross_val_score(RF_model, x, y, cv=10, scoring='f1_weighted')
print('CV F1-score:', fw.mean(), fw.std())
roc_auc = ms.cross_val_score(RF_model, x, y, cv=10, scoring='roc_auc')
print('CV AUC:', roc_auc.mean(), roc_auc.std())

#----------------------------------------------------------------#
#                     训练集10倍交叉验证ROC曲线                     #
#----------------------------------------------------------------#
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内（以上两行代码必须放在画图代码的最前面，否则无效）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold

# Import some data to play with
y = df["Target"]
x = df.loc[:, "SCC": "超氧歧化酶"]
X, y = x[y != 2], y[y != 2]
n_samples, n_features = X.shape

#数据标准化
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
X = standardScaler.fit_transform(x)  #为交叉验证做数据预处理

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=10)
classifier = RandomForestClassifier(oob_score=True, random_state=17)
classifier.fit(X, y)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(dpi=300,figsize=(5,5)) #设置图片的大小和分辨率
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=1.5, color="r", label="Chance", alpha=0.8) #绘制随机概率的虚线

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=1.5,
    alpha=0.8,)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
)

#plt.title("Ten-fold cross-validated ROC curve of RF",fontsize=10,fontproperties="Times New Roman")
font = {'family': 'Times New Roman','size': 6,} #设置图例格式，方便下面用
plt.legend(loc="lower right",prop=font) #添加图例，位置在“下右”
plt.xlim([-0.03, 1.0])    #设置x，y轴的刻度范围
plt.ylim([0.0, 1.03])
plt.xticks(size=10,fontproperties="Times New Roman") #设置x，y轴的刻度标签的大小和字体
plt.yticks(size=10,fontproperties="Times New Roman")
plt.ylabel('Sensitivity',fontsize=10,fontproperties="Times New Roman")
plt.xlabel('1-Specificity',fontsize=10,fontproperties="Times New Roman")
plt.rcParams['figure.dpi'] = 150                #分辨率
plt.rcParams['savefig.dpi'] = 600               #设置保存图像的dpi
plt.show()



