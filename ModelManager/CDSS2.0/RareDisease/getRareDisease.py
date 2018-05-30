#coding:utf-8
import json
import csv
import codecs
from itertools import islice
def getRareDisease(rareFile,Path='/home/jq/quanke_data/BYSY/data/train_data2'):
    bingliFile=[]
    diseaseFile=[]
    merge_disease=json.load(open(rareFile, 'r',encoding='utf-8'))
    
    for i in range(8):
        i = i + 1
        bingliList=json.load(open('/home/jq/quanke_data/BJCYYY/' + 'data/train_data2/bingli_feature' + str(i) + '.txt', 'r'))
        diseaseCsv=csv.reader(open('/home/jq/quanke_data/BJCYYY/' +'data/train_data2/diagnosis_name'+str(i)+'.csv'))
        print('i:',i)
        for disease in islice(diseaseCsv, 1, None):
            if disease[2]!='' and disease[2] in merge_disease:
                bingliFile.append(bingliList[int(disease[4])])
                diseaseFile.append(merge_disease[disease[2]])
    print('朝阳医院:',len(diseaseFile))
    

    for i in range(2):
        i = i + 1
        bingliList=json.load(open(Path+'/bingli_feature' + str(i) + '.txt', 'r'))
        diseaseCsv=csv.reader(open(Path+'/diagnosis_name'+str(i)+'.csv'))
        for disease in islice(diseaseCsv, 1, None):
            if disease[2]!='' and disease[2] in merge_disease:
                bingliFile.append(bingliList[int(disease[4])])
                diseaseFile.append(merge_disease[disease[2]])
    assert len(bingliFile) == len(diseaseFile), '病例与疾病名字数目不同'
    print('病例与疾病名字数目:',len(diseaseFile))
    save2libSVMFile(bingliFile,diseaseFile)

def save2libSVMFile(bingliFile,diseaseFile,path='/home/jq/jeeker/DATA_2/RareDisease'):
    from  sklearn.feature_extraction import DictVectorizer
    from  sklearn.preprocessing import LabelEncoder
        #path='/home/jq/jeeker/mergeData_425'
    XArr=bingliFile
    yArr=diseaseFile
    v = DictVectorizer()
    X_v = v.fit_transform(XArr)
    print(X_v.shape,type(X_v))
    import pickle
    pickle.dump(v,open(path+'/X_transform.pickle','wb'),-1)

    json.dump(v.get_feature_names(), codecs.open(path+'/FeatureNameArr.txt', 'w+', encoding='utf-8'), ensure_ascii=False)
    v1 = LabelEncoder()
    Y_v = v1.fit_transform(yArr)
    Y_label = v1.classes_
    json.dump(Y_label.tolist(), codecs.open(path+'/Y_label.txt', 'w+', encoding='utf-8'), ensure_ascii=False)
    print('len(v1.classes_)', len(v1.classes_))
    print(type(Y_v))
    from collections import Counter
    Y_statis=sorted(Counter(Y_v).items())
    print(Y_statis,len(Y_statis))
    #json.dump(list(Y_statis), open(path+'/Y_statis.txt', 'w+', encoding='utf-8'), ensure_ascii=False)
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    X_v = imp.fit_transform(X_v)

    from sklearn.datasets import dump_svmlight_file
    dump_svmlight_file(X_v,Y_v,path+'/total.libsvm')
    
    rareDic={}
    for index in range(0,X_v.shape[0]):
        li=rareDic.get(Y_v[index],list([]))
        li.append(X_v[index])
        rareDic[Y_v[index]]=li
    #print(rareDic)
    #json.dump(rareDic, open(path+'/rare_bingli.txt', 'w+'))
    return rareDic


if __name__=='__main__':
    getRareDisease('./rare_disease_name.txt')
