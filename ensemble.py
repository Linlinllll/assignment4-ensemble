from __future__ import print_function

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler as scaler
from sklearn import preprocessing

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

#数据预处理：标准化
def standarlizer(X_train):
    return preprocessing.scale(X_train)

#取数据
train_x, train_y = fetch_data('./train.csv')
test_x = fetch_data('./test.csv',False)

#画出相关性图
g = sns.heatmap(train_x[["SL","Time","BP","Circulation","HR","EEG"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.show()

#通过相关性分析去除特征
train_x.drop("EEG",1)
test_x.drop("Attribute6",1)

# #k折交叉验证
kfold = StratifiedKFold(n_splits=10)

#使用不同的模型预测数据
def model_compare():
    random_state = 2
    classifiers = []
    classifiers.append(SVC(gamma=0.0000001, kernel='rbf', C=1000))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(BaggingClassifier(base_estimator=DecisionTreeClassifier()))

    cv_results = []
    for classifier in classifiers :
        cv_results.append(cross_val_score(classifier, train_x, train_y, scoring = "accuracy", cv = kfold, n_jobs=4))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
    "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","BaggingClassifier"]})

    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    plt.show()


#网格搜索ExtraTrees参数
def gs_ET():
    ExtC = ExtraTreesClassifier()

    ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 6],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

    gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsExtC.fit(train_x, train_y)

    ExtC_best = gsExtC.best_estimator_

    print(gsExtC.best_score_)

#网格搜索得到的最好参数
ExtC_best = ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='gini',
           max_depth=None, max_features=3, max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=3,
           min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=None,
           oob_score=False, random_state=None, verbose=0, warm_start=False)


# 网格搜索RandomForestClassifier参数
def gs_RFC():
    RFC = RandomForestClassifier()

    rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 6],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


    gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsRFC.fit(train_x, train_y)

    RFC_best = gsRFC.best_estimator_

    print(RFC_best)

    print(gsRFC.best_score_)

#网格搜索得到的最好模型
RFC_best = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features=1, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


#网格搜索BaggingClassifier参数
def gs_bag():
    bag = BaggingClassifier()

    bag_param_grid = {
              "base_estimator": [DecisionTreeClassifier()],
              "max_features": [1, 3, 6],
              "max_samples": [1, 2, 3, 10],
              "bootstrap": [True],
              "n_estimators" :[10, 100, 300],
              "verbose": [0, 10, 100],
              "bootstrap_features": [False,True],
              "oob_score" : [False,True],
              "warm_start" : [False]
    }

    gsBag = GridSearchCV(bag,param_grid = bag_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsBag.fit(train_x, train_y)

    bag_best = gsBag.best_estimator_

    print(bag_best)

    print(gsBag.best_score_)

#网格搜索最好参数
BaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False,random_state=None,
                splitter='best'), bootstrap=True, bootstrap_features=False, max_features=6,max_samples=10, n_estimators=300,
                n_jobs=None, oob_score=False, random_state=None, verbose=10, warm_start=False)

#投票函数结合多个模型
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best)], voting='soft', n_jobs=4)

#数据标准化
standarlizer(train_x)
standarlizer(test_x)

#预测测集标签输出成csv文件
votingC.fit(train_x,train_y)
pred_y = votingC.predict(test_x)
write_data(pred_y,"predict")
