#coding:utf-8

import sys
sys.path.append('/home/jq/jeeker')
import pickle
import json

#数据存放路径
DATAPATH='/home/jq/jeeker/DATA_2.1/mix_288_add'
#模型存放路径
MODELPATH='/home/jq/jeeker/Model_2.1/NB'

#这个函数的作用是数据归一化
def normalX(trainFile):
    from sklearn.datasets import load_svmlight_file
    x_train,y_train=load_svmlight_file(trainFile)

    print(x_train.shape,y_train.shape)
    
    #统计每个疾病的数量，以便观察
    from collections import Counter
    Y_statis=sorted(Counter(y_train).items())
    print(Y_statis,len(Y_statis))
    json.dump(Y_statis, open(DATAPATH+'/Y_statis.txt', 'w+', encoding='utf-8'), ensure_ascii=False)
    
    #归一化数据，并且保存归一化模型，方便线上部署加载使用
    from sklearn import preprocessing
    max_abs_scaler = preprocessing.MaxAbsScaler()
    x_train = max_abs_scaler.fit_transform(x_train)
    pickle.dump(max_abs_scaler, open(DATAPATH+'/MaxAbsScaler.pickle', 'wb'), -1)

    #划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, x_test, Y_train, y_test = train_test_split(x_train,y_train, test_size=0.2)
    from sklearn.datasets import dump_svmlight_file
    dump_svmlight_file(x_test,y_test,DATAPATH+'/Normal_valiation.libsvm')
    dump_svmlight_file(x_train,y_train,DATAPATH+'/Normal_total.libsvm')
    dump_svmlight_file(X_train,Y_train,DATAPATH+'/Normal_train.libsvm')
    
def trainModel(trainFile,testFile):
    import numpy as np
    import TT_ML.data_helper.data_prepare as dataProcess
    import TT_ML.singleLearning.NB as NB
     
    data_helper=dataProcess.DataPrepare()
    #加载训练集和测试集
    x_train,y_train,x_test,y_test=data_helper.loadLibSVMFile(trainFile,testFile)
    #设置模型的参数，调参
    clf = NB.NB_normal(x_train,y_train,x_test,y_test,modelName='BernoulliNB')
    #训练模型
    #clf.fit(x_train,y_train)
    print('TrainModel Start!')
    #保存模型,线上部署需要
    pickle.dump(clf, open(MODELPATH+'/NB_zhuyuan.pickle', 'wb'), -1)
    print('TrainModel Finish!')
    return clf
def testModel(testFile):
    import json
    import pickle
    #加载模型
    model=pickle.load(open(MODELPATH+'/NB_zhuyuan.pickle','rb'))
    import time
    start=time.time()
    from sklearn.datasets import load_svmlight_file
    x_val,Y_test=load_svmlight_file(testFile)
    #x_val=x_val[:100]
    #Y_test=Y_test[:100]
    #得到预测概率的矩阵，我们后期需要对概率矩阵排序，然后返回topN
    proba=model.predict_proba(x_val)

    predict=model.predict(x_val)


    print(proba,proba.shape)
    #print(predict[:20],Y_test[:20])
    classes=model.classes_
    classes=list(classes)
    #print(classes)    
    import numpy as np

    #概率矩阵的排序，从大到小
    nMax = np.argsort(-proba)
    # big->small
    #sortProba = np.array([proba[line_id, i] for line_id, i in enumerate(np.argsort(-proba, axis=1))])
    #Y = json.load(codecs.open('./Y_label.txt', 'r', encoding='utf-8'))
    print('$' * 10)
    top1=0
    top3 = 0
    top5=0
    top8=0
    top10=0

    top1Dic={}
    top3Dic={}
    top5Dic={}
    top8Dic={}
    top10Dic={}

    import csv
    csvfile=open('./predict_DiseaseTop10.csv','w',newline='',encoding='utf-8')
    writer=csv.writer(csvfile)
    writer.writerow(["标签", "疾病名字", "top10"])
    diseaseNameLabel=json.load(open(DATAPATH+'/Y_label.txt','r',encoding='utf-8'))
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
        
        if Y_test[index] in Nindex[:1]:
            top1Dic[Y_test[index]]=top1Dic.get(Y_test[index],0)+1
            top1+=1
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
    dic['top1']=top1/float(sampleN)
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
    #diseaseNameLabel=json.load(open('/home/jq/jeeker/cy_zy_410_955/newYLabel.txt','r',encoding='utf-8'))
    #diseaseNameDic=json.load(open('/home/jq/jeeker/cy_zy_raw328/y_statis.txt','r',encoding='utf-8'))
    #
    from collections import Counter
    Y_statis=sorted(Counter(Y_test).items())
    print(Y_statis)
    import csv
    csvfile=open('./DiseasePredicted_325.csv','w',newline='',encoding='utf-8')
    writer=csv.writer(csvfile)
    writer.writerow(["标签", "疾病名字","top1","top3","top5","top8","top10","该类别实际样本数","该类别训练采样数"])
    for tup in Y_statis:
        try:
            writer.writerow([tup[0],diseaseNameLabel[int(tup[0])],
                                  str(top1Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top1Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',
                                  str(top3Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top3Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',
                                  str(top5Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top5Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',
                                  str(top8Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top8Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',
                                  str(top10Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top10Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')'
                                  ])
#str(top10Dic.get(tup[0],0)/float(tup[1]))[:4]+'('+str(top10Dic.get(int(tup[0]),0))+'/'+str(tup[1])+')',diseaseNameDic[str(int(tup[0]))],300])
        except:
            pass     

    return Y_test,predict,classes


#函数作用是，得到混淆矩阵
def plot_confusion_matrix(y_true, y_pred, labels):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cmap = plt.cm.get_cmap('Accent_r')
    # cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    intFlag = 1# 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                #这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('疾病模型预测混淆矩阵结果', fontproperties=custom_font)
    plt.colorbar()
    # xy = range(0)
    # z = xy
    # sc = plt.scatter(z,z,c=z)
    # plt.colorbar(sc)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel(u'滚动轴承真实类别', fontproperties=custom_font)
    plt.xlabel(u'滚动轴承预测类别', fontproperties=custom_font)
    plt.savefig('./confusion_matrix.jpg', dpi=300)
    #plt.show()

    

if __name__=='__main__':
    #normalX('/home/jq/jeeker/DATA_2.1/mix_288_add/total.libsvm')
    
    #trainModel('/home/jq/jeeker/DATA_2.1/mix_288_add/Normal_train.libsvm','/home/jq/jeeker/DATA_2.1/mix_288_add/Normal_valiation.libsvm')
    testModel('/home/jq/jeeker/DATA_2.1/mix_288_add/Normal_valiation.libsvm')
    #plot_confusion_matrix(y_true, y_pred, labels)
