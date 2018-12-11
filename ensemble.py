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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier


from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
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
#k折交叉验证
kfold = StratifiedKFold(n_splits=10)
# #使用不同的模型
# random_state = 2
# classifiers = []
# classifiers.append(SVC(gamma=0.0000001, kernel='rbf', C=1000))
# classifiers.append(DecisionTreeClassifier(random_state=random_state))
# classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
# classifiers.append(RandomForestClassifier(random_state=random_state))
# classifiers.append(ExtraTreesClassifier(random_state=random_state))
# classifiers.append(GradientBoostingClassifier(random_state=random_state))
# classifiers.append(MLPClassifier(random_state=random_state))
# classifiers.append(KNeighborsClassifier())
# classifiers.append(BaggingClassifier(base_estimator=DecisionTreeClassifier()))
#
# cv_results = []
# for classifier in classifiers :
#     cv_results.append(cross_val_score(classifier, train_x, train_y, scoring = "accuracy", cv = kfold, n_jobs=4))
#
# cv_means = []
# cv_std = []
# for cv_result in cv_results:
#     cv_means.append(cv_result.mean())
#     cv_std.append(cv_result.std())
#
# cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
# "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","BaggingClassifier"]})
#
# g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
# g.set_xlabel("Mean Accuracy")
# g = g.set_title("Cross validation scores")
# plt.show()


# #ExtraTrees
# ExtC = ExtraTreesClassifier()
#
#
# ## Search grid for optimal parameters
# ex_param_grid = {"max_depth": [None],
#               "max_features": [1, 3, 6],
#               "min_samples_split": [2, 3, 10],
#               "min_samples_leaf": [1, 3, 10],
#               "bootstrap": [False],
#               "n_estimators" :[100,300],
#               "criterion": ["gini"]}
#
#
# gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
#
# gsExtC.fit(train_x, train_y)
#
# ExtC_best = gsExtC.best_estimator_
#
# # Best score
# print(gsExtC.best_score_)
#
ExtC_best = ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='gini',
           max_depth=None, max_features=3, max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=3,
           min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=None,
           oob_score=False, random_state=None, verbose=0, warm_start=False)


# # RFC Parameters tunning
# RFC = RandomForestClassifier()
#
#
# ## Search grid for optimal parameters
# rf_param_grid = {"max_depth": [None],
#               "max_features": [1, 3, 6],
#               "min_samples_split": [2, 3, 10],
#               "min_samples_leaf": [1, 3, 10],
#               "bootstrap": [False],
#               "n_estimators" :[100,300],
#               "criterion": ["gini"]}
#
#
# gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
#
# gsRFC.fit(train_x, train_y)
#
# RFC_best = gsRFC.best_estimator_
#
# print(RFC_best)
# # Best score
# print(gsRFC.best_score_)

RFC_best = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features=1, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

Bag_best = BaggingClassifier(base_estimator=DecisionTreeClassifier(), bootstrap=True, bootstrap_features=False,
                             max_features=1.0, max_samples=1.0, n_estimators=10, n_jobs=1, oob_score=False)

#outlier_detction(train_x)
#quantiletransform(train_x,test_x)
#predict()

votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),('bag', Bag_best)], voting='soft', n_jobs=4)
standarlizer(train_x)
standarlizer(test_x)
votingC.fit(train_x,train_y)
pred_y = votingC.predict(test_x)
write_data(pred_y,"predict")
