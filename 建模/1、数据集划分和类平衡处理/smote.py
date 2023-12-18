import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_excel("Cervical.xlsx")
X_whole = data.drop('Target', axis=1)#获取全部特征数据
y_whole = data.Target                #获取全部标签数据
x_train_w, x_test_w, y_train_w, y_test_w = \
    train_test_split(X_whole, y_whole, test_size = 0.3, random_state = 0)
#其中x_train_w：特征数据中70%为训练特征数据，x_test_w：特征数据中30%为测试数据
#其中y_train_w：标签数据中70%的训练标签数据，y_test_w：标签数据中30%为测试数据

from imblearn.over_sampling import SMOTE
oversampler = SMOTE(random_state=0)#数据对象
os_x_train, os_y_train = oversampler.fit_resample(x_train_w, y_train_w)#
print('jieshu')

'''数据保存为excel文件'''
data_train = pd.concat([os_y_train,os_x_train],axis=1)
data_test = pd.concat([y_test_w,x_test_w],axis=1)
data_train.to_excel('训练数据集.xlsx', index=False)
data_test.to_excel('测试数据集.xlsx', index=False)


from xgboost import XGBRegressor as XGBR

