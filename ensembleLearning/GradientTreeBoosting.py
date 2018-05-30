#coding:utf-8
'''
模块设计目的:梯度树提升算法
@author:Jeeker
'''
#算法说明
'''
GDTB 是对于任意的可微损失函数的提升算法的泛化。
GBRT 梯度提升回归树，既能用于分类问题也能用于回归问题。

GBRT优点：
    1.对混合型数据的自然处理（异构特征）。
    2.强大的预测能力。
    3.在输出空间中对异常点的鲁棒性（通过具有鲁棒性的损失函数实现）

GBRT缺点：
    1.由于提升算法的有序性，下一步的结果依赖于上一步，因此难于并行。
'''

#用于分类
'''
GradientBoostingClassifier 既支持二分类又支持多分类问题

超过两类的分类问题需要在每一次迭代时推导 n_classes 个回归树。因此，所有的需要推导的树数量等于 n_classes * n_estimators 。
对于拥有大量类别的数据集我们强烈推荐使用 RandomForestClassifier 来代替 GradientBoostingClassifier 。 
'''
from sklearn.ensemble import GradientBoostingClassifier

def GDBT_normal(X,y):
    clf=GradientBoostingClassifier(n_estimators=100)
    clf=clf.fit(X,y)
    return clf

#用于回归
'''
对于回归问题 GradientBoostingRegressor 支持一系列 different loss functions ，
这些损失函数可以通过参数 loss 来指定；对于回归问题默认的损失函数是最小二乘损失函数（ 'ls' ）。

以下是目前支持的损失函数,具体损失函数可以通过参数 loss 指定:
回归 (Regression)
        1.Least squares ( 'ls' ): 由于其优越的计算性能,该损失函数成为回归算法中的自然选择。 初始模型 通过目标值的均值给出。
        2.Least absolute deviation ( 'lad' ): 回归中具有鲁棒性的损失函数,初始模型通过目 标值的中值给出。
        3.Huber ( 'huber' ): 回归中另一个具有鲁棒性的损失函数,它是最小二乘和最小绝对偏差两者的结合. 其利用 alpha 来控制模型对于异常点的敏感度(详细介绍请参考 [F2001]).
        4.Quantile ( 'quantile' ): 分位数回归损失函数.用 0 < alpha < 1 来指定分位数这个损 失函数可以用来产生预测间隔。（详见 Prediction Intervals for Gradient Boosting Regression ）。
分类 (Classification)
        1.Binomial deviance ('deviance'): 对于二分类问题(提供概率估计)即负的二项 log 似然损失函数。模型以 log 的比值比来初始化。
        2.Multinomial deviance ('deviance'): 对于多分类问题的负的多项log似然损失函数具有 n_classes 个互斥的类。提供概率估计。 初始模型由每个类的先验概率给出.在每一次迭代中 n_classes 回归树被构建,这使得 GBRT 在处理多类别数据集时相当低效。
        3.Exponential loss ('exponential'): 与 AdaBoostClassifier 具有相同的损失函数。与 'deviance' 相比，对被错误标记的样本的鲁棒性较差，仅用于在二分类问题。
'''
from sklearn.ensemble import GradientBoostingRegressor
def GDBT_Regression(X,y):
    clf=GradientBoostingRegressor(loss='ls')
    clf=clf.fit(X,y)
    return clf

#投票分类器
'''
VotingClassifier （投票分类器）的原理是结合了多个不同的机器学习分类器,并且采用多数表决（majority vote）（硬投票） 
或者平均预测概率（软投票）的方式来预测分类标签。 这样的分类器可以用于一组同样表现良好的模型,以便平衡它们各自的弱点。
'''

#voting='hard' 参数设置投票分类器为多数表决方式
def Voting_Hard(X,y):
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

    for clf, label in zip([clf1, clf2, clf3, eclf],
                          ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

#加权平均概率 （软投票）
        # 具体的权重可以通过权重参数 weights 分配给每个分类器。
        #  当提供权重参数 weights 时，收集每个分类器的预测分类概率， 乘以分类器权重并取平均值。然后将具有最高平均概率的类别标签确定为最终类别标签。
def Voting_Soft(X,y):
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    clf1 = LogisticRegression(random_state=1).fit(X,y)
    clf2 = RandomForestClassifier(random_state=1).fit(X,y)
    clf3 = GaussianNB().fit(X,y)
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

#投票器在网格搜索中的应用
'''
1.为了通过预测的类别概率来预测类别标签(投票分类器中的 scikit-learn estimators 必须支持 predict_proba 方法):
2.也可以为单个分类器提供权重
'''
def Voting_UseCV(X,y):
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    clf1=LogisticRegression()
    clf2=RandomForestClassifier()
    clf3=GaussianNB()
    eclf=VotingClassifier(estimators=[('lr',clf1),('rf',clf2),('gnb',clf3)],voting='soft')
    params={'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],}
    grid=GridSearchCV(estimator=eclf,param_grid=params,cv=5)
    grid=grid.fit(X,y)

    #也可以为单个分类器提供权重
    #eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[2, 5, 1])
