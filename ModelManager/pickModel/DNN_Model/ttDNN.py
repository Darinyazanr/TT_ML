#coding:utf-8

import sys
sys.path.append('/home/jq/jeeker')
import pickle

#Keras_MLP


#Keras 下训练DNN算法模型
def trainModel(trainFile,testFile):
    import TT_ML.deepLearning.keras_nn as KerasNN
    import TT_ML.data_helper.data_prepare as dataProcess
    data_helper=dataProcess.DataPrepare()
    #加载模型
    x_train,y_train,x_test,y_test=data_helper.loadLibSVMFile(trainFile,testFile)
    

    clf = KerasNN.Kmodel_1(x_train,y_train,x_test,y_test)
    return clf

#sklearn_MLP
#sklearn 库下训练DNN算法模型
def trainModel_MLP(trainFile,testFile):
    import TT_ML.deepLearning.sklearn_DNN  as sklearnDNN
    import TT_ML.data_helper.data_prepare as dataProcess
    data_helper=dataProcess.DataPrepare()
    x_train,y_train,x_test,y_test=data_helper.loadLibSVMFile(trainFile,testFile)

    clf =  sklearnDNN.MLP_normal(x_train,y_train,x_test,y_test)
    pickle.dump(clf, open('./SklearnDNN_zhuyuan.pickle', 'wb'), -1)
    return clf

def getPredictTopN():
    import TT_ML.ModelStatis.topN as topN
    modelFile='./SklearnDNN_zhuyuan.pickle'
    import TT_ML.data_helper.data_prepare as dataProcess
    data_helper=dataProcess.DataPrepare()
    x_train,y_train,x_test,y_test=data_helper.loadLibSVMFile('/home/jq/jeeker/zy_322/train.libsvm','/home/jq/jeeker/zy_322/test.libsvm')
    topN.predictTop(modelFile, x_test, y_test, len(y_test))

def loadModel_Kears():
    from keras.models import load_model
    model=load_model('my_model.h5')
    import TT_ML.data_helper.data_prepare as dataProcess
    data_helper=dataProcess.DataPrepare()
    x_train,y_train,x_test,y_test=data_helper.loadLibSVMFile('/home/jq/jeeker/zy_data2/train.libsvm','/home/jq/jeeker/zy_data2/test.libsvm')
    y_pred = model.predict(x_train[:60000], batch_size=256)
    import numpy as np
    note_prediction=np.argmax(y_pred,axis=1)
    print('*'*20,note_prediction,'*'*20)

    from sklearn.metrics import classification_report, confusion_matrix
    #print(confusion_matrix(y_test, note_prediction))
    print(classification_report(y_train[:60000], note_prediction))
   

def testModel(testFile):
    import json
    import pickle
    model=pickle.load(open('./SklearnDNN_zhuyuan.pickle','rb'))
    import time
    start=time.time()
    from sklearn.datasets import load_svmlight_file
    x_val,Y_test=load_svmlight_file(testFile)
    proba=model.predict_proba(x_val)
    print(proba,proba.shape)
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
        print('-'*20,index)
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
    #
    diseaseNameLabel=json.load(open('/home/jq/jeeker/zy_322/Y_label.txt','r',encoding='utf-8'))
    diseaseNameDic=json.load(open('/home/jq/jeeker/zy_322/y_statis.txt','r',encoding='utf-8'))
    #
    from collections import Counter
    Y_statis=sorted(Counter(Y_test).items())
    print(Y_statis)
    import csv
    csvfile=open('./DiseasePredicted_325.csv','w',newline='')
    writer=csv.writer(csvfile)
    writer.writerow(["标签", "疾病名字", "top3","top5","top8","top10","该类别实际样本数","该类别训练采样数"])
    for tup in Y_statis:
        writer.writerow([tup[0],diseaseNameLabel[int(tup[0])],str(top3Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top3Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',
                                   str(top5Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top5Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',
                                   str(top8Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top8Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',
                                   str(top10Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top10Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',diseaseNameDic[str(int(tup[0]))],300])


if __name__=='__main__':
    #trainModel('/home/jq/jeeker/DATA_2.1/mix_288_add/Normal_train.libsvm','/home/jq/jeeker/DATA_2.1/mix_288_add/Normal_valiation.libsvm')
    trainModel_MLP('/home/jq/jeeker/DATA_2.1/mix_288_add/Normal_train.libsvm','/home/jq/jeeker/DATA_2.1/mix_288_add/Normal_valiation.libsvm')
    testModel('/home/jq/jeeker/DATA_2.1/mix_288_add/Normal_valiation.libsvm')
    #getPredictTopN()
    #loadModel_Kears()
