from sklearn.datasets import load_iris  # flower database
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

#
class MyClassifier:
    # MyClassifier.fit(x_train, y_train)
    #
    # Parameters
    #   x_train:    Training data
    #   y_train:    Training labels
    #
    # Description
    #   loads all the training data into the classifier
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    # MyClassifier.predict(data)
    #
    # Parameters
    #   data    list of data to predict
    #
    # Description
    #   Returns a prediction based on the given data
    def predict(self, data):
        predictions = []  # return array

        # loop through the data and add the label of the nearest
        # training data point to the return array
        for row in data:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    # MyClassifier.closest(row)
    #
    # Parameters
    #   row:    test data to find the closest training data to
    #
    # Description
    #   Given a test data point, find the closest training data point
    def closest(self, row):
        best_dist = distance.euclidean(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = distance.euclidean(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


# ########################################################
# ##################End MyClassifier######################
##########################################################

iris = load_iris()  # load iris database

# load database
x = iris.data
y = iris.target

# split the data int training data and test data
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

# create and train classifier
clf = MyClassifier()
clf.fit(xTrain, yTrain)

# predict
prediction = clf.predict(xTest)
print(accuracy_score(yTest, prediction))
