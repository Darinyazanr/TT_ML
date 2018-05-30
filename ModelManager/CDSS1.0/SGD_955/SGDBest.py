#coding:utf-8

import sys
sys.path.append('/home/jq/jeeker')
import pickle

def normalX(trainFile):
    from sklearn.datasets import load_svmlight_file
    x_train,y_train=load_svmlight_file(trainFile)
    
    from sklearn import preprocessing
    max_abs_scaler = preprocessing.MaxAbsScaler()
    x_train = max_abs_scaler.fit_transform(x_train)
    pickle.dump(max_abs_scaler, open('./MaxAbsScaler.pickle', 'wb'), -1)

    from sklearn.model_selection import train_test_split
    X_train, x_test, Y_train, y_test = train_test_split(x_train,y_train, test_size=0.2)
    from sklearn.datasets import dump_svmlight_file
    dump_svmlight_file(x_test,y_test,'/home/jq/jeeker/cy_zy_410/Normal_valiation.libsvm')
    dump_svmlight_file(x_train,y_train,'/home/jq/jeeker/cy_zy_410/Normal_total.libsvm')
    dump_svmlight_file(X_train,Y_train,'/home/jq/jeeker/cy_zy_410/Normal_train.libsvm')

def trainModel(trainFile,testFile):
    import numpy as np
    import TT_ML.data_helper.data_prepare as dataProcess
    data_helper=dataProcess.DataPrepare()
    x_train,y_train,x_test,y_test=data_helper.loadLibSVMFile(trainFile,testFile)
    from sklearn.linear_model import SGDClassifier
    clf=SGDClassifier(loss='log',penalty='l2')#modified_huber
    clf.partial_fit(x_train,y_train,classes=np.unique(y_train))
    
    #from sklearn.neighbors import KNeighborsClassifier
    #clf=KNeighborsClassifier(n_neighbors=10)
    clf.fit(x_train,y_train)
    print('TrainModel Start!')
    pickle.dump(clf, open('./SGD_zhuyuan.pickle', 'wb'), -1)
    print('TrainModel Finish!')
    return clf

def testModel(testFile):
    import json
    import pickle
    model=pickle.load(open('./SGD_zhuyuan.pickle','rb'))
    import time
    start=time.time()
    from sklearn.datasets import load_svmlight_file
    x_val,Y_test=load_svmlight_file(testFile)
    #x_val=x_val[:100]
    #Y_test=Y_test[:100]
    print('Start Prediction')
    proba=model.predict_proba(x_val)

    predict=model.predict(x_val)
    print(proba,proba.shape)
    #print(predict[:20],Y_test[:20])
    classes=model.classes_
    classes=list(classes)
    #print(classes)    
    import numpy as np
    nMax = np.argsort(-proba)
    # big->small
    #sortProba = np.array([proba[line_id, i] for line_id, i in enumerate(np.argsort(-proba, axis=1))])
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

    import csv
    csvfile=open('./predict_DiseaseTop10.csv','w',newline='',encoding='utf-8')
    writer=csv.writer(csvfile)
    writer.writerow(["标签", "疾病名字", "top10"])
    diseaseNameLabel=json.load(open('/home/jq/jeeker/cy_zy_410_955/newYLabel.txt','r',encoding='utf-8'))
    for index in range(len(Y_test)):
        print(index)
        Maxindex = nMax[index]
        
        Nindex=[]
        for ii in Maxindex:   
            Nindex.append(classes[ii])
        print('-'*20,index)
        #print(Y_test[index],Nindex[:10])
        tmpDisease=[]
        for di in Nindex[:10]:
            tmpDisease.append(diseaseNameLabel[int(di)])
        writer.writerow([int(Y_test[index]),diseaseNameLabel[int(Y_test[index])],' '.join(tmpDisease)])

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
    json.dump(dic, open('./predictTopN.txt', 'w+', encoding='utf-8'), ensure_ascii=False)
    print('*'*20)
    print('Predict Finish!')
    print('*'*20)
    #
    diseaseNameLabel=json.load(open('/home/jq/jeeker/cy_zy_410_955/newYLabel.txt','r',encoding='utf-8'))
    #diseaseNameDic=json.load(open('/home/jq/jeeker/cy_zy_raw328/y_statis.txt','r',encoding='utf-8'))
    #
    from collections import Counter
    Y_statis=sorted(Counter(Y_test).items())
    print(Y_statis)
    import csv
    csvfile=open('./DiseasePredicted_325.csv','w',newline='',encoding='utf-8')
    writer=csv.writer(csvfile)
    writer.writerow(["标签", "疾病名字", "top3","top5","top8","top10","该类别实际样本数","该类别训练采样数"])
    for tup in Y_statis:
        try:
            writer.writerow([tup[0],diseaseNameLabel[int(tup[0])],str(top3Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top3Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',
                                   str(top5Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top5Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',
                                   str(top8Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top8Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')'])
#str(top10Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top10Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',diseaseNameDic[str(int(tup[0]))],300])
        except:
            pass     




if __name__=='__main__':
    #normalX('/home/jq/jeeker/cy_zy_410/caiyangto500.libsvm')
    
    trainModel('/home/jq/jeeker/cy_zy_410_955/Normal_train.libsvm','/home/jq/jeeker/cy_zy_410_955/Normal_valiation.libsvm')
    testModel('/home/jq/jeeker/cy_zy_410_955/Normal_valiation.libsvm')
    
