#coding:utf-8
'''
模块设计目的：SVM实现回归
@author:Jeeker
'''
#算法说明
'''
支持向量分类的方法可以被扩展用作解决回归问题. 这个方法被称作支持向量回归.
支持向量分类生成的模型(如前描述)只依赖于训练集的子集,因为构建模型的 cost function 不在乎边缘之外的训练点. 
类似的,支持向量回归生成的模型只依赖于训练集的子集, 因为构建模型的 cost function 忽略任何接近于模型预测的训练数据.

支持向量分类有三种不同的实现形式: SVR, NuSVR 和 LinearSVR. 
在只考虑线性核的情况下, LinearSVR 比 SVR 提供一个更快的实现形式, 然而比起 SVR 和 LinearSVR, NuSVR 实现一个稍微不同的构思(formulation)
'''
from sklearn import svm
def SVMR_normal(X,y):
    #X = [[0, 0], [2, 2]]
    #y = [0.5, 2.5]
    clf=svm.SVR()
    clf.fit(X,y)
    '''
    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    '''
