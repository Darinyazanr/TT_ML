#coding:utf-8
import time
import pickle
import json

def predictTop(modelFile, X_test, Y_test, sampleN):
    clf = pickle.load(open(modelFile, 'rb'))
    start = time.time()

    #answer = clf.predict(X_test[:sampleN])
    proba = clf.predict_proba(X_test[:sampleN])

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
    for index in range(sampleN):
        #print(index)
        Nindex = nMax[index]
        Npro = sortProba[index]
        if Y_test[index] in Nindex[:3]:
            top3+=1
        elif Y_test[index] in Nindex[:5]:
            top5+=1
        elif Y_test[index] in Nindex[:8]:
            top8+=1
        elif Y_test[index] in Nindex[:10]:
            top10+=1
    dic={}
    end = time.time()
    dic['totalSample']=sampleN
    dic['top3']=top3/float(sampleN)
    dic['top5']=(top3+top5)/float(sampleN)
    dic['top8']=(top3+top5+top8)/float(sampleN)
    dic['top10']=(top3+top5+top8+top10)/float(sampleN)
    dic['TakeTime']=end-start
    json.dump(dic, open('./predictTopN.txt', 'w+', encoding='utf-8'), ensure_ascii=False)
    note_prediction = list(clf.predict(X_test))
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(Y_test, note_prediction))
    print(classification_report(Y_test, note_prediction))
    print('*'*20)
    print('Predict Finish!')
    print('*'*20)
