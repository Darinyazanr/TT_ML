#coding:utf-8
'''
模块设计目的:数据不均衡问题处理,此文件是原始版本，具体请看imBanlance_v2.0.py
@author:Jeeker
'''
def reduceData(X,y,MinNum=10,MaxNum=1000):
    from collections import Counter
    import numpy as np
    x=sorted(Counter(y).items())
    #print(x)
    deleArr=[]
    crapArr=[]
    for tup in x:
        if tup[1]<=MinNum:
            deleArr.append(tup[0])
        elif tup[1]>=MaxNum:
            crapArr.append(tup[0])
    #reduce num <=10
    print('deleArr',len(deleArr))
    print('*'*20)
    print('crapArr',len(crapArr))
    from scipy.sparse import csr_matrix,vstack
    newX=X[0]
    newY=[]
    for index in range(1,len(y)):
        if y[index] not in deleArr:
            newY.append(y[index])
            newX=vstack([newX,X[index]])
    newX = newX[1:]
    print(type(newX),newX.shape)

    #reduce num>=3000
    crapDic = {}
    newY2=[]
    newX2=newX[0]
    for index in range(1, len(newY)):
        if newY[index] in crapArr:
            crapDic[newY[index]] = crapDic.get(newY[index], 0)+1
            if crapDic[newY[index]]<=MaxNum:
                newY2.append(newY[index])
                newX2 = vstack([newX2, newX[index]])
        else:
            newY2.append(newY[index])
            newX2 = vstack([newX2, newX[index]])
    print(sorted(Counter(newY2).items()))
    newY2=np.array(newY2)
    newX2=newX2[1:]
    print(type(newX2),newX2.shape,newY2.shape)
    print(len(np.unique(newY2)))
    return newX2,newY2

def OverSampling_RandomOver(X,y):
    from collections import Counter
    print(sorted(Counter(y).items()))
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    newX, newY = ros.fit_sample(X, y)
    print(newX.shape, newY.shape,type(newX))
    print(sorted(Counter(newY).items()))
    print('-'*20)
    #from imblearn.datasets import make_imbalance
    #newX,newY=make_imbalance(newX,newY,ratio=ratio_multiplier)
    #print(sorted(Counter(newY).items()))
    return newX,newY

def ratio_multiplier(y,ratio=0.5):
    target_stats = Counter(y)
    for key, value in target_stats.items():
        target_stats[key] = int(value * ratio)
    return target_stats

def OverSampling_SMOTE(X,y):
    from collections import Counter
    print(sorted(Counter(y).items()))
    from imblearn.over_sampling import SMOTE
    newX,newY=SMOTE(k_neighbors=3).fit_sample(X,y)
    print(newX.shape, newY.shape)
    print(sorted(Counter(newY).items()))
    return newX, newY

def OverSampling_ADASYN(X,y):
    from collections import Counter
    print(sorted(Counter(y).items()))
    from imblearn.over_sampling import ADASYN
    newX,newY=ADASYN().fit_sample(X,y)
    print(newX.shape, newY.shape)
    print(sorted(Counter(newY).items()))
    return newX, newY

if __name__=='__main__':
    from collections import Counter
    from sklearn.datasets import make_classification

    X, y = make_classification(n_classes=4, class_sep=2,
    weights = [0.1, 0.2,0.3,0.4], n_informative = 3, n_redundant = 1, flip_y = 0,
    n_features = 20, n_clusters_per_class = 1, n_samples = 1000, random_state = 10)

    print('Original dataset shape {}'.format(Counter(y)))

    #OverSampling_RandomOver(X,y)
    OverSampling_SMOTE(X,y)
