#coding:utf-8
'''
模块设计目的:随机森林
@author:Jeeker
'''
'''
skealrn.ensemble 模块包含两个基于 随机决策树 的平均算法：RandomForest算法和Extra-Trees算法。
这两种算法都是专门为树而设计的扰动和组合技术：这种技术通过在分类器构造过程中引用随机性来创建一组不同的分类器。
集成分类器的预测结果就是单个分类器预测结果加一起后的平均值

默认参数下模型复杂度是：O(M*N*log(N)) ， 其中 M 是树的数目， N 是样本数。 
可以通过设置以下参数来降低模型复杂度： min_samples_split , min_samples_leaf , max_leaf_nodes`` 和 ``max_depth 。
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

def RF_normal(x_train,y_train):
    clf=RandomForestClassifier(n_estimators=70,warm_start=True,class_weight='balanced')
    clf=clf.fit(x_train,y_train)
    '''
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
    '''
    
    return clf

'''
在随机森林中（参见 ExtraTreesClassifier 和 ExtraTreesRegressor 类）， 集成模型中的每棵树构建时的样本都是由训练集经过有放回抽样得来的
（例如，自助采样法-bootstrap sample，这里采用西瓜书中的译法）。 
另外，在构建树的过程中进行结点分割时，选择的分割点不再是所有特征中最佳分割点，而是特征的一个随机子集中的最佳分割点。 
由于这种随机性，森林的偏差通常会有略微的增大（相对于单个非随机树的偏差），但是由于取了平均，其方差也会减小，通常能够补偿偏差的增加，
从而产生一个总体上更好的模型。
与原始文献 [B2001] 不同的是，scikit-learn 的实现是取每个分类器预测概率的平均，而不是让每个分类器对类别进行投票。

在极限随机树中（参见 ExtraTreesClassifier 和 ExtraTreesRegressor 类)， 计算分割点方法中的随机性进一步增强。
 在随机森林中，使用的特征是候选特征的随机子集；
 不同于寻找最具有区分度的阈值， 这里的阈值是针对每个候选特征随机生成的，并且选择这些随机生成的阈值中的最佳者作为分割规则。
  这种做法通常能够减少一点模型的方差，代价则是略微地增大偏差：
  
  另外，请注意，在随机森林中，默认使用自助采样法（bootstrap = True）， 然而 extra-trees 的默认策略是使用整个数据集（bootstrap = False）。 
  当使用自助采样法方法抽样时，泛化精度是可以通过剩余的或者袋外的样本来估算的，设置 oob_score = True 即可实现。
'''
def extraTreeClassifier(X,y):
    clf=ExtraTreesClassifier(n_estimators=10,n_jobs=-1,bootstrap=True)# 如果设置 n_jobs = k ，
    clf=clf.fit(X,y)                                      # 则计算被划分为 k 个作业，并运行在机器的 k 个核上。 如果设置 n_jobs = -1 ，则使用机器的所有核。
    return clf
