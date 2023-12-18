# 导入库
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, \
    roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取内部的Excel数据
filename = ("训练数据集-有SMOTE.xlsx")
df = pd.read_excel(filename)

# 将特征赋值给自变量x，将标签赋值给因变量y
y = df["Target"]
x = df.loc[:, "SCC": "超氧歧化酶"]

# 读取外部验证的Excel数据
filename = ("测试数据集.xlsx")
df = pd.read_excel(filename)

# 将特征赋值给自变量x，将标签赋值给因变量y
y_out = df["Target"]
X_out = df.loc[:, "SCC": "超氧歧化酶"]

#数据标准化
standardScaler = StandardScaler()
x = standardScaler.fit_transform(x)  #为交叉验证做数据预处理
X_out = standardScaler.fit_transform(X_out)

# 绘制roc曲线
def calculate_auc(y, pred):
    print("auc:", roc_auc_score(y, pred))
    fpr, tpr, thersholds = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k-', label='ROC (area = {0:.2f})'.format(roc_auc), color='blue', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()

# 使用Yooden法寻找最佳阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

# 计算roc值
def ROC(label, y_prob):
    fpr, tpr, thresholds = roc_curve(label, y_prob)
    roc_auc = auc(fpr, tpr)
    optimal_threshold, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_threshold, optimal_point

# 计算混淆矩阵
def calculate_metric(label, y_prob, optimal_threshold):
    p = []
    for i in y_prob:
        if i >= optimal_threshold:
            p.append(1)
        else:
            p.append(0)
    confusion = confusion_matrix(label, p)
    print(confusion)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    Accuracy = (TP + TN) / float(TP + TN + FP + FN)
    Sensitivity = TP / float(TP + FN)
    Specificity = TN / float(TN + FP)
    return Accuracy, Sensitivity, Specificity

# 多模型比较：
models = [('AdaBoost', AdaBoostClassifier(n_estimators=150, random_state=0)),
          ('GNB', GaussianNB()),
          ('LR', LogisticRegression()),
          ('RF', RandomForestClassifier(oob_score=True, random_state=17)),
          ('SVM', SVC(probability=True, kernel='linear', class_weight='balanced')),
          ('XGBoost', XGBClassifier(objective='binary:logistic', learning_rate=None, max_depth=None, min_child_weight=None, reg_lambda=None))
          ]

# 循环训练模型
results = []
roc_ = []
for name, model in models:
    clf = model.fit(x, y)
    pred_proba = clf.predict_proba(X_out)
    y_prob = pred_proba[:, 1]
    fpr, tpr, roc_auc, Optimal_threshold, optimal_point = ROC(y_out, y_prob)
    Accuracy, Sensitivity, Specificity = calculate_metric(y_out, y_prob, Optimal_threshold)
    result = [Optimal_threshold, Accuracy, Sensitivity, Specificity, roc_auc, name]
    results.append(result)
    roc_.append([fpr, tpr, roc_auc, name])

df_result = pd.DataFrame(results)
df_result.columns = ["Optimal_threshold", "Accuracy", "Sensitivity", "Specificity", "AUC_ROC", "Model_name"]
print(df_result)
#--------------------------------------------------------------#
#                     绘制多组对比roc曲线                         #
#--------------------------------------------------------------#
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内（以上两行代码必须放在画图代码的最前面，否则无效）
lw = 1.2     #设置线宽
fig  = plt.subplots(dpi=300,figsize=(5,5)) #设置图片的大小和分辨率
plt.plot(roc_[0][0], roc_[0][1], color='seagreen', lw=lw, label=roc_[0][3] + ' (AUC = %0.3f)' % roc_[0][2])  #海绿色
plt.plot(roc_[1][0], roc_[1][1], color='orangered', lw=lw, label=roc_[1][3] + ' (AUC = %0.3f)' % roc_[1][2]) #橘红
plt.plot(roc_[2][0], roc_[2][1], color='slateblue', lw=lw, label=roc_[2][3] + ' (AUC = %0.3f)' % roc_[2][2]) #岩蓝
plt.plot(roc_[3][0], roc_[3][1], color='deeppink', lw=lw, label=roc_[3][3] + ' (AUC = %0.3f)' % roc_[3][2])  #深粉
plt.plot(roc_[4][0], roc_[4][1], color='olivedrab', lw=lw, label=roc_[4][3] + ' (AUC = %0.3f)' % roc_[4][2]) #橄榄绿
plt.plot(roc_[5][0], roc_[5][1], color='orange', lw=lw, label=roc_[5][3] + ' (AUC = %0.3f)' % roc_[5][2])    #橘黄色
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')

#plt.title("Receiver Operator Characteristic Curves", fontsize=10, fontproperties="Times New Roman")
font = {'family': 'Times New Roman','size': 8}
plt.legend(loc="lower right", fontsize=8, prop=font) #添加图例，位置在“下右”
plt.xlim([-0.03, 1.0]) #设置x，y轴的刻度范围
plt.ylim([0.0, 1.03])
plt.xticks(size=10,fontproperties="Times New Roman") #设置x，y轴的刻度标签的大小和字体
plt.yticks(size=10,fontproperties="Times New Roman")
plt.ylabel('Sensitivity',fontsize=10,fontproperties="Times New Roman")
plt.xlabel('1-Specificity', fontsize=10, fontproperties="Times New Roman")
plt.rcParams['figure.dpi'] = 150                #分辨率
plt.rcParams['savefig.dpi'] = 600               #设置保存图像的dpi
plt.show()


