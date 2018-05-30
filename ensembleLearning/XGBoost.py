#coding:utf-8
'''
模块设计目的： xgboost算法实现
@author:Jeeker
'''
#算法说明
'''
Most of parameters in xgboost are about bias variance tradeoff.

该算法避免过拟合方法：
1.除了算法本身带正则化之外
2.可以减少树的深度，最小叶子的权重。
3.增加随机性
'''

'''
The XGBoost python module is able to load data from:
libsvm txt format file
Numpy 2D array, and
xgboost binary buffer file.
'''
import xgboost as xgb

def XGB_normal(trainFile,testFile):

    dtrain=xgb.DMatrix(trainFile)
    dtest=xgb.DMatrix(testFile)
    y_test=dtest.get_label()
    import numpy as np
    a=dtrain.get_label()
    NumClass=len(np.unique(np.array(a)))
    '''    
    __init__(self, data, label=None, missing=None, weight=None, silent=False, feature_names=None, feature_types=None, nthread=None)
           Data matrix used in XGBoost.
    data : string/numpy array/scipy.sparse/pd.DataFrame
           Data source of DMatrix.
           When data is string type, it represents the path libsvm format txt file,
           or binary file that xgboost can read from.
           label : list or numpy 1-D array, optional
           Label of the training data.
           missing : float, optional
           Value in the data which needs to be present as a missing value. If
           None, defaults to np.nan.
           weight : list or numpy 1-D array , optional
           Weight for each instance.
           silent : boolean, optional
           Whether print messages during construction
           feature_names : list, optional
           Set names for features.
           feature_types : list, optional
           Set types for features.
    '''
    param = {'max_depth': 6,'eta': 0.1,'silent': 1,'objective': 'multi:softprob','num_class':NumClass}
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round=5
    bst=xgb.train(params=param,dtrain=dtrain,num_boost_round=num_round,evals=evallist)
    '''
      
    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : DMatrix
        Data to be trained.
    num_boost_round: int
        Number of boosting iterations.
    evals: list of pairs (DMatrix, string)
        List of items to be evaluated during training, this allows user to watch
        performance on the validation set.
    obj : function
        Customized objective function.
    feval : function
        Customized evaluation function.
    maximize : bool
        Whether to maximize feval.
    early_stopping_rounds: int
        Activates early stopping. Validation error needs to decrease at least
        every <early_stopping_rounds> round(s) to continue training.
        Requires at least one item in evals.
        If there's more than one, will use the last.
        Returns the model from the last iteration (not the best one).
        If early stopping occurs, the model will have three additional fields:
        bst.best_score, bst.best_iteration and bst.best_ntree_limit.
        (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
        and/or num_class appears in the parameters)
    evals_result: dict
        This dictionary stores the evaluation results of all the items in watchlist.
        Example: with a watchlist containing [(dtest,'eval'), (dtrain,'train')] and
        a parameter containing ('eval_metric': 'logloss')
        Returns: {'train': {'logloss': ['0.48253', '0.35953']},
                  'eval': {'logloss': ['0.480385', '0.357756']}}
    verbose_eval : bool or int
        Requires at least one item in evals.
        If `verbose_eval` is True then the evaluation metric on the validation set is
        printed at each boosting stage.
        If `verbose_eval` is an integer then the evaluation metric on the validation set
        is printed at every given `verbose_eval` boosting stage. The last boosting stage
        / the boosting stage found by using `early_stopping_rounds` is also printed.
        Example: with verbose_eval=4 and at least one item in evals, an evaluation metric
        is printed every 4 boosting stages, instead of every boosting stage.
    learning_rates: list or function (deprecated - use callback API instead)
        List of learning rate for each boosting round
        or a customized function that calculates eta in terms of
        current number of round and the total number of boosting round (e.g. yields
        learning rate decay)
    xgb_model : file name of stored xgb model or 'Booster' instance
        Xgb model to be loaded before training (allows training continuation).
    callbacks : list of callback functions
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using xgb.callback module.
        Example: [xgb.callback.reset_learning_rate(custom_rates)]
    
    Returns
    -------
    booster : a trained booster model

    '''
    bst.save_model('cy_zy_322.model')
    '''
    # dump model
    bst.dump_model('dump.raw.txt')
    # dump model with feature map
    bst.dump_model('dump.raw.txt', 'featmap.txt')
    
    bst.load_model('model.bin')  # load model
    '''
    #预测
    ypred = bst.predict(dtest)
    print('@'*20,ypred)
    
    #print('predicting, classification error=%f' % (
    #    sum(int(ypred[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test))))
    return bst

'''
Plotting
You can use plotting module to plot importance and output tree.
To plot importance, use plot_importance. This function requires matplotlib to be installed.
xgb.plot_importance(bst)
To plot the output tree via matplotlib, use plot_tree, specifying the ordinal number of the target tree. This function requires graphviz and matplotlib.
xgb.plot_tree(bst, num_trees=2)
When you use IPython, you can use the to_graphviz function, which converts the target tree to a graphviz instance. The graphviz instance is automatically rendered in IPython.
xgb.to_graphviz(bst, num_trees=2)
'''
