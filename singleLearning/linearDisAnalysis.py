#coding:utf-8
'''
#模块设计目的：线性和二次判别分析

@author:Jeeker
'''
#使用线性判别分析来降维
'''
通过把输入的数据投影到由最大化类之间分离的方向所组成的线性子空间，可以执行有监督降维。
输出的维度必然会比原来的类别数量更少的。因此它总体而言是十分强大的降维方式，同样也仅仅在多分类环境下才能感觉到。
关于维度的数量可以通过 n_components 参数来调节。 值得注意的是，这个参数不会对 discriminant_analysis.LinearDiscriminantAnalysis.fit 
或者 discriminant_analysis.LinearDiscriminantAnalysis.predict 产生影响。
'''

'''
数学推导：
具体地说，对于线性以及二次判别分析，  被建模成密度多变量高斯分布。
为了把该模型作为分类器使用，我们只需要从训练数据中估计出类的先验概率  P(y=k)（通过每个类 k的实例的比例得到） 类别均值 uk （通过经验样本的类别均值得到）以及
协方差矩阵（通过经验样本的类别协方差或者正则化的估计器 estimator 得到: 见下面的 shrinkage ）。

Shrinkage（收缩）:
收缩是一种在训练样本数量相比特征而言很小的情况下可以提升的协方差矩阵预测（准确性）的工具。 在这个情况下，经验样本协方差是一个很差的预测器。
收缩 LDA 可以通过设置 discriminant_analysis.LinearDiscriminantAnalysis 类的 shrinkage 参数为 ‘auto’ 来实现。
shrinkage parameter （收缩参数）的值同样也可以手动被设置为 0-1 之间。特别地，0 值对应着没有收缩（这意味着经验协方差矩阵将会被使用），
 而 1 值则对应着完全使用收缩（意味着方差的对角矩阵将被当作协方差矩阵的估计）。
 设置该参数在两个极端值之间会估计一个（特定的）协方差矩阵的收缩形式
 
 默认的 solver 是 ‘svd’。它可以进行classification (分类) 以及 transform (转换),而且它不会依赖于协方差矩阵的计算（结果）。这在特征数量特别大的时候十分具有优势。然而，’svd’ solver 无法与 shrinkage （收缩）同时使用。
‘lsqr’ solver 则是一个高效的算法，它仅用于分类使用。它支持 shrinkage （收缩）。
 
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def LDA(X,y):
    lda = LinearDiscriminantAnalysis(n_components=2,solver="svd", store_covariance=True,)
    clf = lda.fit(X, y)
    return clf
def QDA(X,y):
    qda = QuadraticDiscriminantAnalysis(store_covariances=True)
    clf = qda.fit(X, y)
    return clf

def LDA_shrink(X,y):
    clf1=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(X,y)
    clf2=LinearDiscriminantAnalysis(solver='lsqr',shrinkage=None).fit(X,y)
    return clf1,clf2