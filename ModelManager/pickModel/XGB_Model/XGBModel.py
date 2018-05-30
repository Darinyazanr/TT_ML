#coding:utf-8
'''
模块设计目的:XGB 模型
@author:Jeeker
'''
import sys
sys.path.append('/home/jq/jeeker')

#训练模型
def trainModel(trainFile,testFile):
    import TT_ML.ensembleLearning.XGBoost as XGBoost
    #具体可以查看TT_ML.ensembleLearning.XGBoost的调参注释
    clf = XGBoost.XGB_normal(trainFile,testFile)
    return clf

#测试模型
def testModel(testFile):
    import json
    import xgboost as xgb
    x_val=xgb.DMatrix(testFile)
    Y_test=x_val.get_label()
    print(Y_test)
    #加载模型
    model=xgb.Booster(model_file='cy_zy_322.model')
    import time
    start=time.time()

    proba=model.predict(x_val)
    #print(proba)
    import numpy as np
    nMax = np.argsort(-proba)
    # big->small
    sortProba = np.array([proba[line_id, i] for line_id, i in enumerate(np.argsort(-proba, axis=1))])
    #Y = json.load(codecs.open('./Y_label.txt', 'r', encoding='utf-8'))
    print('$' * 10)
    top3 = 0
    top5=0
    top8=0
    top10=0
    
    top3Dic={}
    top5Dic={}
    top8Dic={}
    top10Dic={}
 
    for index in range(len(Y_test)):
        #print(index)
        Nindex = nMax[index]
        Npro = sortProba[index]
        print('-'*20)
        #print(Y_test[index],Nindex[:10])
        if Y_test[index] in Nindex[:3]:
            top3Dic[Y_test[index]]=top3Dic.get(Y_test[index],0)+1
            top3+=1
        if Y_test[index] in Nindex[:5]:
            top5Dic[Y_test[index]]=top5Dic.get(Y_test[index],0)+1
            top5+=1
        if Y_test[index] in Nindex[:8]:
            top8Dic[Y_test[index]]=top8Dic.get(Y_test[index],0)+1
            top8+=1
        if Y_test[index] in Nindex[:10]:
            top10Dic[Y_test[index]]=top10Dic.get(Y_test[index],0)+1
            top10+=1
    dic={}
    end = time.time()
    sampleN=len(Y_test)
    dic['totalSample']=sampleN
    dic['top3']=top3/float(sampleN)
    dic['top5']=(top5)/float(sampleN)
    dic['top8']=(top8)/float(sampleN)
    dic['top10']=(top10)/float(sampleN)
    dic['TakeTime']=end-start
    json.dump(dic, open('./predictTopN_325.txt', 'w+', encoding='utf-8'), ensure_ascii=False)
    print('*'*20)
    print('Predict Finish!')
    print('*'*20)
    diseaseNameLabel=json.load(open('/home/jq/jeeker/DATA_2.1/mix_288_add/Y_label.txt','r',encoding='utf-8'))
    diseaseNameDic=json.load(open('/home/jq/jeeker/DATA_2.1/mix_288_add/Y_statis.txt','r',encoding='utf-8'))
    from collections import Counter
    Y_statis=sorted(Counter(Y_test).items())
    print(Y_statis)
    import csv
    csvfile=open('./DiseasePredicted_325.csv','w',newline='')
    writer=csv.writer(csvfile)
    writer.writerow(["标签", "疾病名字", "top3","top5","top8","top10","该类别实>际样本数","该类别训练采样数"])
    for tup in Y_statis:  
        writer.writerow([tup[0],diseaseNameLabel[int(tup[0])],str(top3Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top3Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',
                                   str(top5Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top5Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',
                                   str(top8Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top8Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',
                                   str(top10Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top10Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',diseaseNameDic[str(int(tup[0]))],300])
    

    
                                     



  



if __name__=='__main__':
    trainModel('/home/jq/jeeker/DATA_2.1/mix_288_add/Normal_train.libsvm','/home/jq/jeeker/DATA_2.1/mix_288_add/Normal_valiation.libsvm')
    testModel('/home/jq/jeeker/DATA_2.1/mix_288_add/Normal_valiation.libsvm')
