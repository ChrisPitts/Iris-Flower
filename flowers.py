import numpy as np
from sklearn.datasets import load_iris  # flower database
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt
#
iris = load_iris()  # load iris database

# load database
x = iris.data
y = iris.target


xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)


# create and train classifier
clf = tree.DecisionTreeClassifier()
clf.fit(xTrain, yTrain)

# predict
prediction = clf.predict(xTest)
print(accuracy_score(yTest, prediction))
plt.matshow(confusion_matrix(yTest, prediction), cmap=plt.cm.gray)
plt.show()
