#coding:utf-8
'''
#模块设计目的：数据预处理

1.数据的读取：
        1.1 从mongoDB数据库直接读取还是 读取一个json文件夹
2.数据的预处理：
        2.1 特征是人工提取还是   需要算法降维（PCA或者LDA）
        2.2 数据的清理（同义词替换，异常词处理）
        2.3 数据归一化
3.类别不平衡问题：
        3.1 对类别较少的疾病进行过采样  SMOTE算法
        3.2 对类别较多的疾病进行欠采样
        3.3 组合/集成方法   对类别较多的疾病 划分子集 类似随机森林
        3.4 代价敏感学习 阈值迁移
            对于分类中不同样本数量的类别分别赋予不同的权重（一般思路分类中的小样本量类别权重高，大样本量类别权重低），然后进行计算和建模。
            使用这种方法时需要对样本本身做额外处理，只需在算法模型的参数中进行相应设置即可。
            很多模型和算法中都有基于类别参数的调整设置，以scikit-learn中的SVM为例，
            通过在class_weight : {dict, 'balanced'}中针对不同类别针对不同的权重，来手动指定不同类别的权重。
            如果使用其默认的方法balanced，那么SVM会将权重设置为与不同类别样本数量呈反比的权重来做自动均衡处理，
            计算公式为：n_samples / (n_classes * np.bincount(y))。

@author:Jeeker
'''
import json
import codecs
import csv
from itertools import islice
import numpy as np

#添加TT_ML模块的路径
import sys
sys.path.append('/home/jq/jeeker')

#添加读取数据文件的路径，此处读取的是北京朝阳医院的数据
PATH='/home/jq/quanke_data/BJCYYY/'
class DataPrepare(object):
    def __init__(self):
        pass
    def reducingDimensions():
        pass
    def readData(self):
        #从txt读文件
        pass
    def clean_data(self):
        pass
    def normalization_data(self):
        pass

    def balance_data(self):
        #某病数量低于某个阈值时，去掉
        #类别均衡
        pass

    # 将多个文件读取
    def multi_compose(self,dept_name):
        bingliFile=[]
        diseaseFile=[]
        for i in range(8):
            i = i + 1
            #读取朝阳医院的病例数据和对应病例的疾病名字
            bingliList=json.load(open(PATH + 'data/train_data2/bingli_feature' + str(i) + '.txt', 'r'))
            diseaseCsv=csv.reader(open(PATH +'data/train_data2/diagnosis_name'+str(i)+'.csv'))

            #打开疾病层次表，此处的作用是：挑选出属于疾病层次表的疾病进行训练和预测
            merge_disease_table ='/home/jq/quanke_modelv2.0/data_analysis/synonyms/疾病层次表5月17日.txt'
            merge_disease=json.load(codecs.open(merge_disease_table, 'r'))
            print('i:',i)
            for disease in islice(diseaseCsv, 1, None):
                if 'all' in dept_name:
                    if disease[2]!='' and disease[2] in merge_disease:
                        bingliFile.append(bingliList[int(disease[4])])
                        diseaseFile.append(merge_disease[disease[2]])
                elif disease[0] in dept_name:
                    bingliFile.append(bingliList[int(disease[4])])
                    diseaseFile.append(disease[2])
        print('朝阳医院:',len(diseaseFile))
        #读取北医三院数据
        for i in range(2):
            i = i + 1
            bingliList=json.load(open('/home/jq/quanke_data/BYSY/data/train_data2/bingli_feature' + str(i) + '.txt', 'r'))
            diseaseCsv=csv.reader(open('/home/jq/quanke_data/BYSY/data/train_data2/diagnosis_name'+str(i)+'.csv'))

            for disease in islice(diseaseCsv, 1, None):
                if 'all' in dept_name:
                    if disease[2]!='' and disease[2] in merge_disease:
                        bingliFile.append(bingliList[int(disease[4])])
                        diseaseFile.append(merge_disease[disease[2]])
                elif disease[0] in dept_name:
                    bingliFile.append(bingliList[int(disease[4])])
                    diseaseFile.append(disease[2])
        assert len(bingliFile) == len(diseaseFile), '病例与疾病名字数目不同'
        print('病例与疾病名字数目:',len(diseaseFile))
        #将病例数据和对应疾病名字进行one-hot转换成矩阵，以便训练
        #将病例对应的疾病名字进行标签化，从0-X作为Label
        #并且保存为libsvm格式
        self.save2libSVMFile(bingliFile,diseaseFile)



    def save2libSVMFile(self,bingliFile,diseaseFile,path='/home/jq/jeeker/DATA_2.1/mix_288_add'):
        from  sklearn.feature_extraction import DictVectorizer
        from  sklearn.preprocessing import LabelEncoder


        XArr=bingliFile
        yArr=diseaseFile
        #对kv形式的特征进行离散化处理
        v = DictVectorizer()
        #将XArr转换成离散化的，存储方式为稀疏矩阵的X_v矩阵
        X_v = v.fit_transform(XArr)
        print(X_v.shape,type(X_v))
        #保存转换文件，作用是：线上部署时，前端过来的kv形式需要转换为矩阵形式，模型才能预测处理
        import pickle
        pickle.dump(v,open(path+'/X_transform.pickle','wb'),-1)
        #保存离散化的特征
        json.dump(v.get_feature_names(), codecs.open(path+'/FeatureNameArr.txt', 'w+', encoding='utf-8'), ensure_ascii=False)
        #将疾病名字标签化
        v1 = LabelEncoder()
        Y_v = v1.fit_transform(yArr)
        Y_label = v1.classes_
        #把转换后的标签对应的疾病名字保存，作用是:线上部署时，模型预测的是标签，需要找到对应的疾病名字.
        json.dump(Y_label.tolist(), codecs.open(path+'/Y_label.txt', 'w+', encoding='utf-8'), ensure_ascii=False)
        print('len(v1.classes_)', len(v1.classes_))
        print(type(Y_v))

        #此处的作用是。在保存为libsvm格式时，没有NAN值的出现，至于为什么有的某行矩阵会出现nan，有待查验
        from sklearn.preprocessing import Imputer
        imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        X_v = imp.fit_transform(X_v)

        #将转换后的，X_v,Y_v保存为libsvm格式，这样训练模型时，可以直接加载使用
        from sklearn.datasets import dump_svmlight_file
        dump_svmlight_file(X_v,Y_v,path+'/total.libsvm')

    #此函数的作用是：加载libsvm文件得到对应的矩阵
    def loadLibSVMFile(self,trainFile,testFile):
        from sklearn.datasets import load_svmlight_file
        x_train,y_train=load_svmlight_file(trainFile)
        x_test,y_test=load_svmlight_file(testFile)
        return x_train,y_train,x_test,y_test    
if __name__=='__main__':
    data_helper=DataPrepare()
    data_helper.multi_compose(['all'])
