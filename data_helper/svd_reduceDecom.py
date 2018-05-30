#coding:utf-8
'''
模块设计目的:对于X数据降维，减少特征

暂时没使用的原因：因为此类降维方式对离散化的类别特征是不太合理的。
@author:Jeeker
'''
def svd_reduce(X,y,n_import=500):
    from sklearn.decomposition import TruncatedSVD
    svd=TruncatedSVD(n_components=n_import)
    X1=svd.fit_transform(X)#return a dense array
    from scipy.sparse import csr_matrix
    X1=csr_matrix(X1)
    print(type(X1),type(X),X1.shape,X.shape)
    return X1,y

def sparse_pca(X,y,n_import=500):
    from sklearn.decomposition import SparsePCA
    spca=SparsePCA(n_components=n_import,random_state=123)
    X1=spca.fit_transform(X)
    print(X1.shape, X.shape)
    return X1, y
def t_sne(X,y,n_import=500):
    from sklearn.manifold import TSNE
    tsne=TSNE(n_components=n_import,random_state=123)
    X1=tsne.fit_transform(X)
    print(X1.shape, X.shape)
    return X1, y
