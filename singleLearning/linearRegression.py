#coding:utf-8
'''
模块设计目的：广义线性模型

@author:Jeeker
'''
from sklearn import linear_model


#普通最小二乘法
def linearRegression_normal(X,y):

    reg=linear_model.LinearRegression()
    reg.fit(X,y)
    print(reg.coef_)#w   O(np^2)
    return reg
#岭回归
def linear_Ridge(X,y):

    clf=linear_model.Ridge(alpha=0.5)
    clf.fit(X, y)
    print(clf.coef_)  # w   O(np^2)

    #设置正则化参数：广义交叉验证  该对象与 GridSearchCV 的使用方法相同，
    # 只是它默认为 Generalized Cross-Validation(广义交叉验证 GCV)，这是一种有效的留一验证方法（LOO-CV）
    clf_CV=linear_model.RidgeCV(alphas=[0.1,1.0,10.0])
    clf_CV.fit(X,y)
    print(clf_CV.coef_,clf_CV.alpha_)

#Lasso回归
#它倾向于使用具有较少参数值的情况，有效地减少给定解决方案所依赖变量的数量
def linear_Lasso(X,y):
    reg=linear_model.Lasso(alpha=0.1)
    reg.fit(X,y)

    #正则化参数
    #scikit-learn 通过交叉验证来公开设置 Lasso alpha 参数的对象: LassoCV and LassoLarsCV。
    # LassoLarsCV 是基于下面解释的 :ref:`least_angle_regression`(最小角度回归)算法。
    #对于具有许多线性回归的高维数据集， LassoCV 最常见。 然而，LassoLarsCV 在寻找 alpha parameter 参数值上更具有优势，
    # 而且如果样本数量与特征数量相比非常小时，通常 LassoLarsCV 比 LassoCV 要快。
    reg_lasso=linear_model.LassoCV(alphas=[0.0,0.1,1.0])
    reg_lasso.fit(X,y)
    print(reg_lasso.alpha_,reg_lasso.coef_)

