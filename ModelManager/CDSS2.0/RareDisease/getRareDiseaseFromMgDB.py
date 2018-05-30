#coding:utf-8
'''
模块设计目的:从mongoDB中获取症状列表和鉴别诊断
@author:Jeeker
'''
#a=['a','d','c']
#a={'d1':['a','b','c'],'d2':['a','d']}
#b=['e','s']


from pymongo import MongoClient
import json



def getHanJianBingSymptom(ip,port):
    dic = json.load(open('/home/jq/jeeker/DATA_2/RareDisease/rare_disease_name.txt', 'r', encoding='utf-8'))
    print(dic, len(dic))
    # '192.168.8.20:20000' ,'20000'
    client = MongoClient(ip, port)
    db = client.medkn
    conn_disease = db.disease_edit

    count = conn_disease.count()
    print('%s中数量为：%d' % ('疾病数目', count))
    disease_All = {}

    for k,v in dic.items():
        everyDisease = []
        print("process disease:", k)
        ryjlDic = conn_disease.find_one({'name': k})
        # print(type(ryjlDic))  #dic
        if ryjlDic:
            print(k)
            try:
                ryjlDic_clear = ryjlDic['obj']
                chiefComplaint = ryjlDic_clear['clinical_manifestation']['symptom_sign']
                try:
                    symptomArr = chiefComplaint['symptom_sign']
                    for i in symptomArr:
                        everyDisease.append(i['symptom_sign_example'])
                except:
                    pass
            except:
                pass
        everyDisease=list(set(everyDisease))
        if everyDisease and everyDisease[0]!='None':
            disease_All[k]=everyDisease
    print(disease_All,len(disease_All))
    json.dump(disease_All, open('/home/jq/jeeker/DATA_2/RareDisease/RareDiseaseSymptom.txt', 'w', encoding='utf-8'), ensure_ascii=False)

    #return disease_All


def symptomDistance(symptomArr,dicArrFile):
    dic=json.load(open(dicArrFile, 'r', encoding='utf-8'))

    dicScore={}

    for k,v in dic.items():

        temp=[0]*len(v)
        baseV=[1]*len(v)
        for index  in range(0,len(v)):
            for iindex in  range(0,len(symptomArr)):
                try:
                    if v[index]==symptomArr[iindex]:
                        temp[index]=1
                except:
                    pass
        print(temp)
        print(baseV)

        jishu=0
        for index in range(len(temp)):
            if temp[index]==baseV[index]:
               jishu+=1
        cos=float(jishu)/len(temp)

        print(cos)
        dicScore[k]=cos
    print(dicScore)
    return dicScore

#{'a':['b','c'],'d':['b','d']}
def differentDignois(disease):
    pass

def bingliDistance(totalSvm):
    from sklearn.datasets import load_svmlight_file
    X,y=load_svmlight_file(totalSvm)

    rareDic={}
    for index in range(0,X.shape[0]):
        li=rareDic.get(y[index],list([]))
        li.append(X[index])
        rareDic[y[index]]=li
    print(rareDic)
    return rareDic

def computeCosine(x,y):
    import numpy as np
    x = x.toarray().reshape(-1)
    y = y.toarray().reshape(-1)

    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos = x.dot(y) / (Lx * Ly)
    if np.isnan(cos):
        cos = -1
    return cos

def predictSimlar(newBingLi,rareDic):

    scoreDic={}
    for k,v in rareDic.items():
        if len(v)==1:
            print('1')
            score=computeCosine(newBingLi,v[0])
            scoreDic[int(k)]=score
        if len(v)==2:
            print('2')
            score=computeCosine(newBingLi,v[0])+computeCosine(newBingLi,v[1])
            scoreDic[int(k)]=score
        if len(v)>2:
            print('3')
            arr=[]
            for i in v:
                #s=spatial.distance.cosine(newBingLi,i)
                s=computeCosine(newBingLi,i)
                arr.append(s)
            #print(arr)
            arr=sorted(arr)
            score=sum(arr[1:-1])/float(len(arr[1:-1]))
            scoreDic[int(k)]=score
    print(scoreDic)
    return scoreDic


def ensembleModels(sympArr,newbingLi,YlabelFile):
    '''
    :param sympArr:['发烧','头疼'] 类似这样的症状列表
    :param newbingLi: 已经被罕见病的Xtransform转码过的一条病例
    :return: 罕见疾病列表排名列表
    '''
    dic1=symptomDistance(sympArr,'/home/jq/jeeker/DATA_2/RareDisease/RareDiseaseSymptom.txt')
    rareDic = bingliDistance('/home/jq/jeeker/DATA_2/RareDisease/total.libsvm')
    dic2=predictSimlar(newbingLi,rareDic=rareDic)
    Y_label=json.load(open(YlabelFile,'r',encoding='utf-8'))
    dic3={}
    for k,v in dic2.items():
        dic3[Y_label[k]]=v
    print(dic1,dic3)


    #策略：将症状相似度和罕见疾病病例相似度  做权衡
    preDic={}
    for k,v in dic1.items():
        if v>=0.5:
            preDic[k]=v*0.5
    for k,v in dic3.items():
        if v>=1e-02:
            preDic[k]=preDic.get(k,0)+v/0.5
    print('-'*20)
    print(preDic)




if __name__=='__main__':

    #from sklearn.datasets import load_svmlight_file
    #X, y = load_svmlight_file('/home/jq/jeeker/DATA_2/RareDisease/total.libsvm')
    #rareDic=bingliDistance('./total.libsvm')
    #predictSimlar(X[7],rareDic)


    getHanJianBingSymptom('192.168.8.31',27017)
    #symptomDistance(["发育不良", "运动障碍"],'./RareDiseaseSymptom.txt')
    #ensembleModels(["心悸",  "无力"],X[971],'/home/jq/jeeker/DATA_2/RareDisease/Y_label.txt')
