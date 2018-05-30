#coding:utf-8
'''
模块设计目的:Lgbm 模型
@author:Jeeker
'''
import sys
sys.path.append('/home/jq/jeeker')
import numpy as np
def trainModel(trainFile,testFile):
    
    import lightgbm as lgb
    #加载数据
    lgb_train=lgb.Dataset(trainFile)
    lgb_eval=lgb.Dataset(testFile)
    #numClass=len(np.unique(lgb_train.get_label()))
    #print('numClass:',numClass)
    #添加参数
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': {'l2', 'auc'},
        'num_class':109,
        #'num_leaves': 500,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    print('Start training...')
    bst = lgb.train(params,
                    lgb_train,
                    num_boost_round=2,
                    valid_sets=lgb_eval)
    #保存模型
    bst.save_model('model.txt')
    #bst = lgb.Booster(model_file='model.txt')  #init model
if __name__=='__main__':
    trainModel('/home/jq/jeeker/xinnei_data/train.libsvm','/home/jq/jeeker/xinnei_data/test.libsvm')
