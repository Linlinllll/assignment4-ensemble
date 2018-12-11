from __future__ import print_function
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler as scaler
from sklearn import preprocessing
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import csv

#取数据
def fetch_data(filename,is_train=True):
    data=pd.read_csv(filename)
    if is_train:
        x=data.iloc[:,[0,1,2,3,4,5]]
        y=data.iloc[:,6]
        return x,y
    return data

#写数据
def write_data(pred_y,filename):
    with open('{}.csv'.format(filename),'w')as w:
        w.write('Id,Category')
        for i ,y in enumerate(pred_y,start=1):
            w.write('\n{},{}'.format(i,int(y)))

#异常点检测：
def outlier_detction(X):
    # fit a Minimum Covariance Determinant (MCD) robust estimator to data
    #robust_cov = MinCovDet().fit(X)
    #y_pred_train = MinCovDet().predict(X)
    # compare estimators learnt from the full data set with true parameters
    #emp_cov = EmpiricalCovariance().fit(X)
    csvFile2 = open('result.csv', 'wb')
    writer = csv.writer(csvFile2)

    clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.000001)
    clf.fit(X)
    y_pred_train = clf.predict(X)
    print(y_pred_train[y_pred_train==-1].size)
    for i in range(10000):
        if y_pred_train[i]!=-1:
           writer.writerow(X[i])


#去掉异常点
def delete(filename,i):
    data = pd.read_csv(filename)
    x = data.iloc[:, [0, 1, 2, 3, 4, 5]]
    x = x.drop([i])

#数据预处理：标准化
def standarlizer(X_train):
    return preprocessing.scale(X_train)

#数据预处理：非线性转化
def quantiletransform(X_train,X_test):
    quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
    X_train = quantile_transformer.fit_transform(X_train)
    X_test = quantile_transformer.transform(X_test)
    np.percentile(X_train[:, 0], [0, 25, 50, 75, 100])

#确定svm模型及参数
def model():
    #return svm.SVC(decision_function_shape='ovo')0.282
    #return svm.SVC(gamma=0.000001, kernel='rbf', C=10)0.666
    #return svm.SVC(gamma=0.000001, kernel='rbf', C=100)
    return svm.SVC(gamma=0.0000001, kernel='rbf', C=1000)
    #return svm.SVC(gamma=0.00000001, kernel='rbf', C=10000)
    #return svm.SVC(gamma='scale',decision_function_shape='ovr')

#预测test数据标签
def predict():
    clf = model()
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)
    write_data(pred_y,"predict")


train_x, train_y = fetch_data('./train.csv')
test_x = fetch_data('./test.csv',False)
#outlier_detction(train_x)
#quantiletransform(train_x,test_x)
#predict()
