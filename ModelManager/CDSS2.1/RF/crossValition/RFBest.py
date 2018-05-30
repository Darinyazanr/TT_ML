#coding:utf-8

import sys
sys.path.append('/home/jq/jeeker')
import pickle
import json

def trainModel(trainFile,testFile):
    import numpy as np
    import TT_ML.data_helper.data_prepare as dataProcess
    data_helper=dataProcess.DataPrepare()
    x_train,y_train,x_test,y_test=data_helper.loadLibSVMFile(trainFile,testFile)
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier(n_estimators=70,class_weight='balanced')
    print('TrainModel Finish!')
    print('交叉验证开始')
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import cross_val_score
   
    cv = ShuffleSplit(n_splits=5, test_size=0.01, random_state=0)
    accs=cross_val_score(clf,x_train,y_train,cv=cv,n_jobs=5)
    print('交叉验证结果:',accs)
    
    print('交叉验证结束')
    return clf


if __name__=='__main__':
    
    trainModel('/home/jq/jeeker/DATA_2.1/mix_288_add/Normal_train.libsvm','/home/jq/jeeker/DATA_2.1/mix_288_add/Normal_valiation.libsvm')
