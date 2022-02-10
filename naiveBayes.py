from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt

def naive_bayes(X_train, y_train, X_test, y_test):
    #create bayes model
    naive_model = ComplementNB().fit(X_train,y_train)
    #use bayes model to predeict
    y_pred = naive_model.predict(X_test)
    #print results
    print(confusion_matrix(y_pred,y_test))
    print(classification_report(y_pred,y_test))

    #plot results
    fpr_dt_1, tpr_dt_1,_ = roc_curve(y_test,y_pred)
    plt.plot(fpr_dt_1,tpr_dt_1,label = "ROC curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.gcf().set_size_inches(5, 5)
    plt.show()
