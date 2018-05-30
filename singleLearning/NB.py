#coding:utf-8
'''
模块设计目的:朴素贝叶斯
GaussianNB(高斯朴素贝叶斯)、MultinomialNB(多项式朴素贝叶斯)、BernoulliNB(伯努利朴素贝叶斯)
@author:Jeeker
'''
def NB_normal(x_train,y_train,modelName='MultinomialNB'):
    import numpy as np
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import BernoulliNB

    if modelName=='MultinomialNB':
        clf = MultinomialNB()
    elif modelName=='GaussianNB':
        clf=GaussianNB()
    elif modelName=='BernoulliNB':
        clf=BernoulliNB()

    import numpy as np
    print(x_train.shape, y_train.shape, type(x_train),type(y_train))
    clf.partial_fit(x_train,y_train,classes=np.unique(y_train))
    #import pickle
    #pickle.dump(clf, open('./NB_zhuyuan.pickle', 'wb'), -1)
    #note_prediction = list(clf.predict(x_test))
    #from sklearn.metrics import classification_report, confusion_matrix
    #print(confusion_matrix(y_test, note_prediction))
    #print(classification_report(y_test, note_prediction))
    return clf

if __name__=='__main__':
    pass
