#coding:utf-8
'''
模块设计目的:
@author:Jeeker
'''
import numpy as np
def MLP_normal(x_train,y_train, x_test,y_test):
    #使用sklearn库包下的DNN算法模型
    from sklearn.neural_network import MLPClassifier
    clf=MLPClassifier(solver='sgd',hidden_layer_sizes=(100,500,100),warm_start=True)
    print(clf.get_params())
    #训练模型
    clf.partial_fit(x_train, y_train, classes=np.unique(y_train))

    return clf

if __name__=='__main__':
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = iris.data
    target = iris.target
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test=train_test_split(data,target,test_size=0.1, random_state=42)
    MLP_normal(X_train, X_test, y_train, y_test)
