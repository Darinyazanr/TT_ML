#coding:utf-8
'''
模块设计目的:
@author:Jeeker
'''
from collections import Counter
import numpy as np

#此函数的作用是设置每个类别采样比率
def ratio_multiplier(y,ratio=50):
    target_stats = Counter(y)
    for key, value in target_stats.items():
        target_stats[key] = int(value * ratio)
    return target_stats

#类别数据不均衡化处理
def imbalanceData(X,y):
    #先统计出每个类别的数量
    tup=sorted(Counter(y).items())
    #分别存不同数量段的类别
    arr_1000=[]
    arr_100 = []
    arr_1 = []
    #此处选取了大于400的进行降采样，小于400大于0的类别进行过采样
    for item in tup:
        if item[1]>400:
            arr_1000.append(item[0])
        elif item[1]<=400 and item[1]>=0:
            arr_100.append(item[0])
        else:
            arr_1.append(item[0])
    print(len(arr_1000),len(arr_100),len(arr_1))

    #因为是X是稀疏矩阵，所以需要对应稀疏矩阵的拼接
    from scipy.sparse import csr_matrix,vstack
    newX_1000=X[0]
    newX_100 = X[0]
    #newX_1 = X[0]

    newY_1000=[]
    newY_100 = []
    #newY_1 = []
    for index in range(1,len(y)):
        if y[index] in arr_1000:
            newX_1000=vstack([newX_1000,X[index]])
            newY_1000.append(y[index])
        if y[index] in arr_100:
            newX_100=vstack([newX_100,X[index]])
            newY_100.append(y[index])
        #if y[index] in arr_1:
        #    newX_1=vstack([newX_1,X[index]])
        #    newY_1.append(y[index])

    #此处将划分大于阈值的类别对应的病例数据和低于阈值对应的病例数据
    newX_1000=newX_1000[1:]
    newX_100 = newX_100[1:]
    #newX_1 = newX_1[1:]

    #大于阈值的下采样到阈值
    from imblearn.under_sampling import RandomUnderSampler
    rus=RandomUnderSampler(random_state=0,replacement=False)
    X_1000,Y_1000=rus.fit_sample(newX_1000,newY_1000)
    print('1000 finish!',X_1000.shape,Y_1000.shape)
    #低于阈值的，合成新样本到阈值
    #此处有三种方法进行过采样,应该分别尝试，测试结果
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import ADASYN
    from imblearn.over_sampling import SMOTE
    smote_enn = SMOTEENN(n_neighbors=5,random_state=0)
    #smote =SMOTE(k_neighbors=2)
    X_100, Y_100 = smote_enn.fit_sample(newX_100, newY_100)
    print('100 finish!',X_100.shape,Y_100.shape)
    '''
    #小于10个的，先重复样本到10个，然后合成采样到1000
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_1, Y_1 = ros.fit_sample(newX_1, newY_1)
    from imblearn.combine import SMOTETomek
    smote_tomek = SMOTETomek(random_state=0,ratio=ratio_multiplier)
    X_1, Y_1 = smote_tomek.fit_sample(X_1, Y_1)
    print('1 finish!',X_1.shape,Y_1.shape)
    '''

    #将过采样和降采样的数据重新拼接，以便后续使用
    resX=vstack([X_1000,X_100])
    import numpy as np
    resY=np.concatenate([Y_1000,Y_100])
    print('all finish')
    print(resX.shape,resY.shape)
    from sklearn.datasets import dump_svmlight_file

    tup=sorted(Counter(resY).items())
    print(tup)
    #保存均衡化的数据
    dump_svmlight_file(resX,resY,'/home/jq/jeeker/DATA_2/sample_mix_249_add/total.libsvm')
    
if __name__=='__main__':
    from sklearn.datasets import load_svmlight_file
    x_train,y_train=load_svmlight_file('/home/jq/jeeker/DATA_2/mix_249_add/total.libsvm')
    print(x_train.shape,y_train.shape)
    imbalanceData(x_train,y_train)
