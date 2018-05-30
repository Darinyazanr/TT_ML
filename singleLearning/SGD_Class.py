#coding:utf-8
'''
模块设计目的:随机梯度下降算法用于分类
@author:Jeeker

2018/01/09
'''
#算法说明
'''
随机梯度下降(SGD) 是一种简单但又非常高效的方法，主要用于凸损失函数下线性分类器的判别式学习，例如(线性) 支持向量机 和 Logistic 回归 。
 尽管 SGD 在机器学习社区已经存在了很长时间, 但是最近在 large-scale learning （大规模学习）方面 SGD 获得了相当大的关注。
 
SGD 已成功应用于在文本分类和自然语言处理中经常遇到的大规模和稀疏的机器学习问题。
对于稀疏数据，本模块的分类器可以轻易的处理超过 10^5 的训练样本和超过 10^5 的特征。

(随机梯度下降法）的优势:
高效。
易于实现 (有大量优化代码的机会)。

（随机梯度下降法）的劣势:
SGD 需要一些超参数，例如 regularization （正则化）参数和 number of iterations （迭代次数）。
SGD 对 feature scaling （特征缩放）敏感。
'''
from sklearn.linear_model import SGDClassifier
def SGD_normal(x_train,y_train,x_test,y_test):
    #X = [[0., 0.], [1., 1.]]
    #y = [0, 1]

    import numpy as np
    print(x_train.shape, y_train.shape, type(x_train))

    clf=SGDClassifier(loss='hinge',penalty='l2')
    clf.partial_fit(x_train,y_train,classes=np.unique(y_train))

    '''
    SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=5, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)
    '''
    #参数说明
    '''
    loss="hinge": (soft-margin) linear Support Vector Machine （（软-间隔）线性支持向量机），
    loss="modified_huber": smoothed hinge loss （平滑的 hinge 损失），
    loss="log": logistic regression （logistic 回归），
    and all regression losses below（以及所有的回归损失）。
    
    默认设置为 penalty="l2" 。 L1 penalty （惩罚）导致稀疏解，使得大多数系数为零。
    Elastic Net（弹性网）解决了在特征高相关时 L1 penalty（惩罚）的一些不足。
    参数 l1_ratio 控制了 L1 和 L2 penalty（惩罚）的 convex combination （凸组合）。
    '''
    note_prediction = list(clf.predict(x_test))
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, note_prediction))
    print(classification_report(y_test, note_prediction))
    return clf

#多分类
'''
SGDClassifier 通过利用 “one versus all” （OVA）方法来组合多个二分类器，从而实现多分类。对于每一个 K 类, 可以训练一个二分类器来区分自身和其他 K-1 个类。
在测试阶段，我们计算每个分类器的 confidence score（置信度分数）（也就是与超平面的距离），并选择置信度最高的分类。
'''
def SGD_multiClass(X,y):
    pass
