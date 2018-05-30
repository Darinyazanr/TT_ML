#coding:utf-8


from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000,n_features=100, n_clusters_per_class=1,n_classes=4)
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_bin=lb.fit_transform(y)

import numpy as np
y_labels=np.ones_like(y_bin)
y_labels[y_bin ==0] = -1
print(y_labels)
from scipy import sparse
sx=sparse.csr_matrix(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sx, y_labels)


def binary_class():
    from fastFM import sgd
    fm=sgd.FMClassification(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
    fm.fit(X_train, y_train)
    y_pred = fm.predict(X_test)
    y_pred_proba = fm.predict_proba(X_test)
    from sklearn.metrics import accuracy_score, roc_auc_score
    print('acc:', accuracy_score(y_test, y_pred))
    print('auc:', roc_auc_score(y_test, y_pred_proba))

from fastFM import als
class FMClassifier(als.FMClassification):
    def fit(self, X, y, *args):
        #y = y.copy()
        #y[y == 0] = -1
        return super(FMClassifier, self).fit(X, y, *args)

    def predict_proba(self, X):
        probs = super(FMClassifier, self).predict_proba(X)
        return np.tile(probs, 2).reshape(2, probs.shape[0]).T


def multi_class():
    from sklearn.multiclass import OneVsRestClassifier
    fm = OneVsRestClassifier(FMClassifier(n_iter=500, random_state=42), n_jobs=-1)
    fm.fit(X_train, y_train)
    y_pred = fm.predict(X_test)
    y_pred_proba = fm.predict_proba(X_test)
    print(y_test)
    print('-'*20)
    print(y_pred)
    print('+'*20)
    print(y_pred_proba)    

def trainModel(trainFile):
    from sklearn.datasets import load_svmlight_file
    X,y=load_svmlight_file(trainFile)
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer(neg_label=-1, pos_label=1)
    y_bin=lb.fit_transform(y)
    print(y_bin)
    print(type(y_bin))
    #y_bin[y_bin==0]=-1
    #print(type(y_bin)) 
    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, y_bin)
    #print('split finish!')
    from sklearn.multiclass import OneVsRestClassifier
    fm = OneVsRestClassifier(FMClassifier(n_iter=500, random_state=42), n_jobs=-1)
    fm.fit(X, y_bin)
    import pickle
    pickle.dump(fm, open('./FM_zhuyuan.pickle', 'wb'), -1)

    y_pred = fm.predict(X[:50])
    y_pred_proba = fm.predict_proba(X[:50])
    print(y[:50])
    print('-'*20)
    print(y_pred)
    print('+'*20)
    print(y_pred_proba)

    
if __name__=='__main__':
    #binary_class()
    #multi_class()
    trainModel('/home/jq/jeeker/cy_zy_raw326/total.libsvm')
