#coding:utf-8
'''
模块设计目的:KNN
@author:Jeeker
'''
def KNN_normal(x_train,y_train,Neighbors=5):
    from sklearn.neighbors import KNeighborsClassifier
    clf=KNeighborsClassifier(n_neighbors=Neighbors)
    clf.fit(x_train,y_train)
    #note_prediction = list(clf.predict(x_test[:3000]))
    #from sklearn.metrics import classification_report, confusion_matrix
    #print(confusion_matrix(y_test[:3000], note_prediction))
    #print(classification_report(y_test[:3000], note_prediction))
    return clf

