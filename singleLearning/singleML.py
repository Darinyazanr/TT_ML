#coding:utf-8

'''
设计目的：实现针对较为简单一类算法一个框架，实现自动化训练，测试，反馈结果。


该模块主要包含了传统的，区别于集成学习算法的，较为简单的算法。
1.借助于使用sklearn机器学习包
2.具有算法的随意添加和删减，伸缩性较强。

time:2018/01/08
'''

from sklearn import discriminant_analysis #判别式分析
'''
discriminant_analysis.LinearClassifierMixin
discriminant_analysis.LinearDiscriminantAnalysis #线性判别模型
discriminant_analysis.QuadraticDiscriminantAnalysis #二次判别分析
'''

from sklearn import svm       #svm
'''
svm.LinearSVC
svm.LinearSVR
svm.NuSVC
svm.NuSVR
svm.OneClassSVM
svm.SVC
svm.SVR
'''
from sklearn import neighbors #近邻算法


from sklearn.cross_validation import train_test_split
class singleML(object):
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3,random_state=123)

    #分类
    def classModel(self):
        self.classModelDic={}
        self.classModelDic['LDA']=discriminant_analysis.LinearDiscriminantAnalysis(solver="svd",store_covariance=True)

        pass

    #回归
    def regressionModel(self):
        self.regressionModelDic={}
        pass

    #聚类
    def clusterModel(self):
        self.clusterModelDic={}
        pass

    #训练
    def trainModel(self,model):
        from sklearn.model_selection import GridSearchCV
        #class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, fit_params=None,
        # n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)

        parameters={}#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 4], 'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}
        clf=GridSearchCV(model,param_grid=parameters,n_jobs=1)
        clf.fit(self.X_train,self.y_train)
        # 输出best score
        print("Best score: %0.3f" % clf.best_score_)
        print("Best parameters set:")
        # 输出最佳的分类器到底使用了怎样的参数
        best_parameters = clf.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        return clf
