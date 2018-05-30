#coding:utf-8
'''
模块设计目的:AdaBoost 算法
@author:Jeeker

2018/01/09
'''

#算法说明
'''
AdaBoost的核心思想是用反复修正数据的权重来训练一系列的弱分类器（一个弱分类器仅仅比随机猜测好一点，比如一个简单的决策树），
由这些弱分类器的预测结果通过加权投票或者加权求和的方式组合，得到最终的预测结果。

在每一次所谓的提升（boosting）迭代中，修改每一个训练样本应用于新一轮学习器的权重。
 初始化时,将所有弱学习器的权重都设置为 1/N ,因此第一次迭代仅仅是通过原始数据训练出一个弱学习器。
 在接下来的 连续迭代中,样本的权重逐个地被修改,学习算法也因此要重新应用这些已经修改的权重。在给定的一个迭代中, 
 那些在上一轮迭代中被预测为错误结果的样本的权重将会被增加，而那些被预测为正确结果的样本的权 重将会被降低。
 随着迭代次数的增加，那些难以预测的样例的影响将会越来越大，每一个随后的弱学习器都将 会被强迫更加关注那些在之前被错误预测的样例 [HTF].
'''
from sklearn.ensemble import AdaBoostClassifier
def AdaBoost_normal(X,y):
    clf=AdaBoostClassifier(n_estimators=100)
    clf=clf.fit(X,y)
    '''
    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=100, random_state=None)
          
    注意算法的选择会由一丢丢的不同：algorithm="SAMME"
    '''
    return clf
