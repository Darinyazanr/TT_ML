#coding:utf-8

'''
设计目的：实现针对集成学习算法一个框架，实现自动化训练，测试，反馈结果。


该模块主要包含了集成学习算法。
1.借助于使用sklearn机器学习包
2.具有算法的随意添加和删减，伸缩性较强。


1.bagging
 from sklearn.ensemble import BaggingClassifier
 from sklearn.neighbors import KNeighborsClassifier
 bagging = BaggingClassifier(KNeighborsClassifier(),
                          max_samples=0.5, max_features=0.5)
2.随机森林
 from sklearn.ensemble import RandomForestClassifier
 clf = RandomForestClassifier(n_estimators=10)

3.AdaBoost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)

4.GBDT
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                 max_depth=1, random_state=0).fit(X_train, y_train)
5.XGBoost

time:2018/01/08
'''
from sklearn.cross_validation import train_test_split

class ensembleML(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    # 分类
    def classModel(self):
        self.classModelDic = {}
        pass

    # 回归
    def regressionModel(self):
        self.regressionModelDic = {}
        pass

    # 聚类
    def clusterModel(self):
        self.clusterModelDic = {}
        pass

    # 训练
    def trainModel(self, model):
        from sklearn.model_selection import GridSearchCV
        # class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, fit_params=None,
        # n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)

        parameters = {}  # parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 4], 'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}
        clf = GridSearchCV(model, param_grid=parameters, n_jobs=1)
        clf.fit(self.X_train, self.y_train)
        return clf