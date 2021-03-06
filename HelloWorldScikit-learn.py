import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

np.random.seed(1)

n_samples_per_class = 100

class0_x0_mean = 10.0
class0_x0_std = 2.0
class0_x1_mean = 20.0
class0_x1_std = 2.0

class1_x0_mean = 20.0
class1_x0_std = 2.0
class1_x1_mean = 10.0
class1_x1_std = 2.0

class2_x0_mean = 30.0
class2_x0_std = 2.0
class2_x1_mean = 0.0
class2_x1_std = 2.0

X = np.zeros((3 * n_samples_per_class, 2), dtype=float)
Y = np.zeros((3 * n_samples_per_class), dtype=int)

X[0:n_samples_per_class, 0] = np.random.randn(n_samples_per_class) * class0_x0_std + class0_x0_mean
X[0:n_samples_per_class, 1] = np.random.randn(n_samples_per_class) * class0_x1_std + class0_x1_mean
X[n_samples_per_class:2 * n_samples_per_class, 0] = np.random.randn(n_samples_per_class) * class1_x0_std + class1_x0_mean
X[n_samples_per_class:2 * n_samples_per_class, 1] = np.random.randn(n_samples_per_class) * class1_x1_std + class1_x1_mean
X[2 * n_samples_per_class:3 * n_samples_per_class, 0] = np.random.randn(n_samples_per_class) * class2_x0_std + class2_x0_mean
X[2 * n_samples_per_class:3 * n_samples_per_class, 1] = np.random.randn(n_samples_per_class) * class2_x1_std + class2_x1_mean

Y[n_samples_per_class:2 * n_samples_per_class] = np.ones((n_samples_per_class), dtype=float)
Y[2 * n_samples_per_class:3 * n_samples_per_class] = 2 * np.ones((n_samples_per_class), dtype=float)

fig = plt.figure(1)
ax = fig.add_subplot(1, 2, 1)
ax.scatter(X[0:n_samples_per_class, 0], X[0:n_samples_per_class, 1], c='r', marker='o')
ax.scatter(X[n_samples_per_class:2 * n_samples_per_class, 0], X[n_samples_per_class:2 * n_samples_per_class, 1], c='g', marker='o')
ax.scatter(X[2 * n_samples_per_class:3 * n_samples_per_class, 0], X[2 * n_samples_per_class:3 * n_samples_per_class, 1], c='b', marker='o')
ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_title("Data")
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

# model = LinearSVC()
# model = KNeighborsClassifier(n_neighbors=3)
# model = DecisionTreeClassifier(max_depth=10)
# model = RandomForestClassifier(n_estimators=1000)
# model = AdaBoostClassifier(n_estimators=1000)
model = MLPClassifier(hidden_layer_sizes=(10, 10), learning_rate_init=0.0001, max_iter=10000)

model.fit(X_train, Y_train)

Y_predict = model.predict(X_test)

cm = metrics.confusion_matrix(Y_test, Y_predict, labels=[0, 1, 2])
print("Confusion Matrix:")
print(cm)

prfs = metrics.precision_recall_fscore_support(Y_test, Y_predict)
print("Precision Recall F-score Support:")
print(prfs)

accuracy = metrics.accuracy_score(Y_test, Y_predict)
print("Accuracy:")
print(accuracy)

cr = metrics.classification_report(Y_test, Y_predict)
print("Classification Report:")
print(cr)
