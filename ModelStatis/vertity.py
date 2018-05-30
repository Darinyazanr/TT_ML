#coding:utf-8
import pickle
import json
import sys
sys.path.append('/home/jq/jeeker')
import time

def transXy(bingliFile,diseaseFile,XTransPickle,modelFile,YLabel):
    Y_test=diseaseFile
    XV=pickle.load(open(XTransPickle,'rb'))
    x_trans=XV.transform(bingliFile)
    #print('*'*50,x_trans[:3])
    #load     
    #y_temp=json.load(open(YMap,'r',encoding='utf-8'))
    y_label=json.load(open(YLabel,'r',encoding='utf-8'))
    #y_dic={}
    #for k,v in y_temp.items():
    #    y_dic[int(k)]=y_label[v]    
    #print(y_dic)

    start=time.time()
    #clf = pickle.load(open(modelFile, 'rb'))    
    
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    x_trans=imp.fit_transform(x_trans)
    import xgboost as xgb
    x_val=xgb.DMatrix(x_trans)
    model=xgb.Booster(model_file=modelFile)
    proba=model.predict(x_val)
    #proba = clf.predict_proba(x_trans)

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
    saveResult=[]
    for index in range(len(diseaseFile)):
        #print(index)
        Nindex = nMax[index]
        Npro = sortProba[index]
        NLabel=[]
        for ii in range(10):
            NLabel.append(y_label[Nindex[ii]])  
        #print(NLabel) 
        saveResult.append({Y_test[index]:NLabel})     
        if Y_test[index] in NLabel[:3]:
            top3+=1
        elif Y_test[index] in NLabel[:5]:
            top5+=1
        elif Y_test[index] in NLabel[:8]:
            top8+=1
        elif Y_test[index] in NLabel[:10]:
            top10+=1
        
    dic={}
    end = time.time()
    sampleN=len(Y_test)
    dic['totalSample']=sampleN
    dic['top3']=top3/float(sampleN)
    dic['top5']=(top3+top5)/float(sampleN)
    dic['top8']=(top3+top5+top8)/float(sampleN)
    dic['top10']=(top3+top5+top8+top10)/float(sampleN)
    dic['TakeTime']=end-start
    json.dump(dic, open('./predictTopN.txt', 'w+', encoding='utf-8'), ensure_ascii=False)
    json.dump(saveResult, open('./predictResult.txt', 'w+', encoding='utf-8'), ensure_ascii=False)
    #note_prediction = list(clf.predict(x_trans))
    #from sklearn.metrics import classification_report, confusion_matrix
    #print(confusion_matrix(Y_test, note_prediction))
    #print(classification_report(Y_test, note_prediction))
    print('*'*20)
    print('Predict Finish!')
    print('*'*20)


def testTopN():
    pass

if __name__=='__main__':
    pass
