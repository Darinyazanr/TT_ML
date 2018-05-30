#coding:utf-8
'''
模块设计目的:使用Bagging方法分类

@author:Jeeker
'''
'''
在集成算法中，bagging 方法会在原始训练集的随机子集上构建一类黑盒估计器的多个实例，然后把这些估计器的预测结果结合起来形成最终的预测结果。

该方法通过在构建模型的过程中引入随机性，来减少基估计器的方差(例如，决策树)。 
在多数情况下，bagging 方法提供了一种非常简单的方式来对单一模型进行改进，而无需修改背后的算法。
'''

'''
bagging 方法有很多种，其主要区别在于随机抽取训练子集的方法不同：
        1.如果抽取的数据集的随机子集是样例的随机子集，我们叫做粘贴 (Pasting) [B1999] 。
        2.如果样例抽取是有放回的，我们称为 Bagging [B1996] 。
        3.如果抽取的数据集的随机子集是特征的随机子集，我们叫做随机子空间 (Random Subspaces) [H1998] 。
        4.最后，如果基估计器构建在对于样本和特征抽取的子集之上时，我们叫做随机补丁 (Random Patches) [LG2012] 。
'''
from sklearn.ensemble import BaggingClassifier
def bagging_normal(X,y):
    from sklearn.neighbors import KNeighborsClassifier
    bagging=BaggingClassifier(KNeighborsClassifier(),max_samples=0.5,max_features=0.5,bootstrap=True,bootstrap_features=True,
                              oob_score=True)
