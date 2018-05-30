#coding:utf-8
'''
模块设计目的:决策树实现分类和回归
@author:Jeeker
'''
'''
其目的是创建一种模型从数据特征中学习简单的决策规则来预测一个目标变量的值。

决策树的优势:
        1.便于理解和解释。树的结构可以可视化出来。
        2.训练需要的数据少。其他机器学习模型通常需要数据规范化，比如构建虚拟变量和移除缺失值,不过请注意，这种模型不支持缺失值。
                    由于训练决策树的数据点的数量导致了决策树的使用开销呈指数分布(训练树模型的时间复杂度是参与训练数据点的对数值)。
        3.能够处理数值型数据和分类数据。其他的技术通常只能用来专门分析某一种变量类型的数据集。详情请参阅算法。
        4.能够处理多路输出的问题。
        5.使用白盒模型。如果某种给定的情况在该模型中是可以观察的，那么就可以轻易的通过布尔逻辑来解释这种情况。相比之下，在黑盒模型中的结果就是很难说明清 楚地。
        6.可以通过数值统计测试来验证该模型。这对事解释验证该模型的可靠性成为可能。
        6.即使该模型假设的结果与真实模型所提供的数据有些违反，其表现依旧良好。
决策树的缺点包括:
        1.决策树模型容易产生一个过于复杂的模型,这样的模型对数据的泛化性能会很差。这就是所谓的过拟合.
                一些策略像剪枝、设置叶节点所需的最小样本数或设置数的最大深度是避免出现 该问题最为有效地方法。
        2.决策树可能是不稳定的，因为数据中的微小变化可能会导致完全不同的树生成。这个问题可以通过决策树的集成来得到缓解
        3.在多方面性能最优和简单化概念的要求下，学习一棵最优决策树通常是一个NP难问题。
                因此，实际的决策树学习算法是基于启发式算法，例如在每个节点进 行局部最优决策的贪心算法。这样的算法不能保证返回全局最优决策树。这个问题可以通过集成学习来训练多棵决策树来缓解,这多棵决策树一般通过对特征和样本有放回的随机采样来生成。
        4.有些概念很难被决策树学习到,因为决策树很难清楚的表述这些概念。例如XOR，奇偶或者复用器的问题。
        
        5.如果某些类在问题中占主导地位会使得创建的决策树有偏差。因此，我们建议在拟合前先对数据集进行平衡。
        
备注：scikit-learn 使用 CART 算法的优化版本
'''

from sklearn import tree

'''
DecisionTreeClassifier 既能用于二分类（其中标签为[-1,1]）也能用于多分类（其中标签为[0,…,k-1]）。
'''
def DT_Class_normal(X,y):
    clf=tree.DecisionTreeClassifier()
    clf=clf.fit(X,y)
    '''
    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
    '''
    return clf

'''
DecisionTreeRegressor 类也可以用来解决回归问题。如在分类设置中，拟合方法将数组X和数组y作为参数，只有在这种情况下，y数组预期才是浮点值
'''
def DT_Regre_normal(X,y):
    #X = [[0, 0], [2, 2]]
    #y = [0.5, 2.5]
    clf=tree.DecisionTreeRegressor()
    clf=clf.fit(X,y)
    '''
    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           presort=False, random_state=None, splitter='best')
    '''
    return clf
