from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def naive_bayes(X_train, X_test, y_train,  y_test):
    #create bayes model
    naive_model = ComplementNB().fit(X_train,y_train)
    #use bayes model to predict
    y_pred = naive_model.predict(X_test)
    #performance metrics
    clf_rprt = classification_report(y_pred,y_test, output_dict=True, digits= 4)
    clf_rprt = pd.DataFrame(clf_rprt).iloc[:3, :5].T
    fpr, tpr, _ = roc_curve(y_test,y_pred)
    #plot results
    f, axes = plt.subplots(3)
    sns.heatmap(confusion_matrix(y_pred,y_test), annot=True, ax=axes[0])
    sns.heatmap(clf_rprt, annot=True, ax=axes[1])
    sns.lineplot(x= fpr, y= tpr, ax=axes[2])
    
    f.suptitle('Performance Results', fontsize=16)

    axes[0].set_title("Confusion Matrix")
    axes[1].set_title("Classification Report")
    axes[2].set_title("ROC Curve")
    axes[2].set_xlabel("False Positive Rate", fontsize = 11)
    axes[2].set_ylabel("True Positive Rate", fontsize = 11)
    plt.show()