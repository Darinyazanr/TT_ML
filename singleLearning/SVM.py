#coding:utf-8
'''
模块设计目的:支持向量机算法

监督学习：分类，回归和异常检测

@author:Jeeker
'''

'''
支持向量机的优势在于:
        1.在高维空间中非常高效.
        2.即使在数据维度比样本数量大的情况下仍然有效.
        3.在决策函数（称为支持向量）中使用训练集的子集,因此它也是高效利用内存的.
        4.通用性: 不同的核函数 核函数 与特定的决策函数一一对应.常见的 kernel

        5.解决非均衡问题
        在 SVC, ，如果分类器的数据不均衡（就是说，很多正例很少负例），设置 class_weight='balanced' 与/或尝试不同的惩罚系数 C 。
        
支持向量机的缺点包括:
        1.如果特征数量比样本数量大得多,在选择核函数时要避免过拟合,
                而且正则化项是非常重要的.
        2.支持向量机不直接提供概率估计,这些都是使用昂贵的五次交叉验算计算的.

数据格式：
在 scikit-learn 中,支持向量机提供 dense(numpy.ndarray ,可以通过 numpy.asarray 进行转换) 和 sparse (任何 scipy.sparse) 样例向量作为输出.
然而,要使用支持向量机来对 sparse 数据作预测,它必须已经拟合这样的数据.
使用C代码的 numpy.ndarray (dense) 或者带有 dtype=float64 的 scipy.sparse.csr_matrix (sparse) 来优化性能.

实现细节：
在底层里，我们使用 libsvm 和 liblinear 去处理所有的计算。这些库都使用了 C 和 Cython 去包装。

'''

#分类
'''
SVC 和 NuSVC 是相似的方法, 但是接受稍许不同的参数设置并且有不同的数学方程.
另一方面, LinearSVC 是另一个实现线性核函数的支持向量分类. 记住 LinearSVC 不接受关键词 kernel, 因为它被假设为线性的. 
它也缺少一些 SVC 和 NuSVC 的成员(members) 比如 support_ .
'''
from sklearn import svm

'''
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)
'''
def SVM_normal(X,y):
    clf=svm.SVC()
    clf.fit(X,y)
    #SVMs 决策函数取决于训练集的一些子集, 称作支持向量. 这些支持向量的部分特性可以在 support_vectors_, support_ 和 n_support 找到
    #获得支持向量
    print(clf.support_vectors_)
    #获得支持向量的索引
    print(clf.support_)
    #每一类别获得支持向量的数量
    print(clf.n_support_)
    return clf


#多元分类
'''
SVC 和 NuSVC 为多元分类实现了 “one-against-one” 的方法 (Knerr et al., 1990) 如果 n_class 是类别的数量, 
那么 n_class * (n_class - 1) / 2 分类器被重构, 而且每一个从两个类别中训练数据.
为了给其他分类器提供一致的交互, decision_function_shape 选项允许聚合 “one-against-one” 分类器的结果成 (n_samples, n_classes) 的大小到
'''
def SVM_multiClass(X,y):
    # X = [[0], [1], [2], [3]]
    # Y = [0, 1, 2, 3]
    clf=svm.SVC(decision_function_shape='ovo')
    clf.fit(X,y)
    '''
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    '''
    clf_linear=svm.LinearSVC()
    clf_linear.fit(X,y)
    '''
    LinearSVC 实现 “one-vs-the-rest” 多类别策略, 从而训练 n 类别的模型. 如果只有两类, 只训练一个模型.
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
    '''

#非均衡问题
'''
    这个问题期望给予某一类或某个别样例能使用的关键词 class_weight 和 sample_weight 提高权重(importance).
    SVC (而不是 NuSVC) 在 fit 方法中生成了一个关键词 class_weight. 
    它是形如 {class_label : value} 的字典, value 是浮点数大于 0 的值, 把类 class_label 的参数 C 设置为 C * value.
'''
def SVM_unbalance(X,y):

    # fit the model and get the separating hyperplane
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, y)

    # fit the model and get the separating hyperplane using weighted classes
    wclf = svm.SVC(kernel='linear', class_weight={1: 10})
    wclf.fit(X, y)

