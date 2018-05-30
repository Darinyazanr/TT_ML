#coding:utf-8
'''
模块设计目的:使用LDA主题模型 尝试找到每个主题下关键词


1.使用sklearn库包
2.使用gensim库包
@author:Jeeker

2018/01/10
'''

#算法说明
'''
在scikit-learn中,LDA主题模型的类在sklearn.decomposition.LatentDirichletAllocation包中，
其算法实现主要基于变分推断EM算法，而没有使用基于Gibbs采样的MCMC算法实现。

scikit-learn除了我们原理篇里讲到的标准的变分推断EM算法外，还实现了另一种在线变分推断EM算法，
它在变分推断EM算法的基础上，为了避免文档内容太多太大而超过内存大小，而提供了分步训练(partial_fit函数)，
即一次训练一小批样本文档，逐步更新模型，最终得到所有文档LDA模型的方法。
'''

#由于LDA是基于词频统计的，因此一般不用TF-IDF来做文档特征。

'''
sklearn 实现
'''

from sklearn.decomposition import LatentDirichletAllocation

def doc2TfArray_sklearn(X):
    from sklearn.feature_extraction.text import CountVectorizer
    cntVector=CountVectorizer()
    cntTf=cntVector.fit_transform(X)
    #print(cntTf)
    #print(cntVector.get_feature_names())
    return cntTf

#这里的X应该是每个文档的词频矩阵
def LDA_sklearn(X,y):
    lda=LatentDirichletAllocation(n_components=len(set(y)),learning_offset=50,random_state=0)
    clf=lda.fit_transform(X)
    print(clf)
    print(lda.components_)
    return lda

'''
gensim实现
'''
dictionary={}

from gensim import corpora
def doc2TfArray_gensim(doc):
    doc_use=[x.split() for x in doc]
    dictionary=corpora.Dictionary(doc_use)
    corpus=[dictionary.doc2bow(x) for x in doc_use]
    print(corpus)
    return corpus,dictionary

import gensim
def LDA_gensim(X,y,dic):
    lda=gensim.models.LdaModel
    ldaModel=lda(corpus=X,num_topics=len(set(y)),id2word=dic)
    print(ldaModel.print_topics(num_topics=-1,num_words=5))




if __name__=='__main__':
    doc=["沙瑞金 赞叹 易学习 的 胸怀 ， 是 金山 的 百姓 有福 ， 可是 这件 事对 李达康 的 触动 很大 。 易学习 又 回忆起 他们 三人 分开 的 前一晚 ， 大家 一起 喝酒 话别 ， 易学习 被 降职 到 道口 县当 县长 ， 王大路 下海经商 ， 李达康 连连 赔礼道歉 ， 觉得 对不起 大家 ， 他 最 对不起 的 是 王大路 ， 就 和 易学习 一起 给 王大路 凑 了 5 万块 钱 ， 王大路 自己 东挪西撮 了 5 万块 ， 开始 下海经商 。 没想到 后来 王大路 竟然 做 得 风生水 起 。 沙瑞金 觉得 他们 三人 ， 在 困难 时期 还 能 以沫 相助 ， 很 不 容易 。",
         "沙瑞金 向 毛娅 打听 他们 家 在 京州 的 别墅 ， 毛娅 笑 着 说 ， 王大路 事业有成 之后 ， 要 给 欧阳 菁 和 她 公司 的 股权 ， 她们 没有 要 ， 王大路 就 在 京州 帝豪园 买 了 三套 别墅 ， 可是 李达康 和 易学习 都 不要 ， 这些 房子 都 在 王大路 的 名下 ， 欧阳 菁 好像 去 住 过 ， 毛娅 不想 去 ， 她 觉得 房子 太大 很 浪费 ， 自己 家住 得 就 很 踏实 。",
         "347 年 （ 永和 三年 ） 三月 ， 桓温 兵至 彭模 （ 今 四川 彭山 东南 ） ， 留下 参军 周楚 、 孙盛 看守 辎重 ， 自己 亲率 步兵 直攻 成都 。 同月 ， 成汉 将领 李福 袭击 彭模 ， 结果 被 孙盛 等 人 击退 ； 而 桓温 三 战三胜 ， 一直 逼近 成都 。"]
    #t=doc2TfArray_sklearn(doc)
    #LDA_sklearn(t,[0,0,1])

    cor,dic=doc2TfArray_gensim(doc)
    LDA_gensim(cor,[0,0,1],dic)