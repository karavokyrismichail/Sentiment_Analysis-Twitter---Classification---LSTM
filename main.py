from dataProcess import getData
from naiveBayes import naive_bayes

X_train, X_test, y_train, y_test = getData()

naive_bayes(X_train, X_test, y_train, y_test)