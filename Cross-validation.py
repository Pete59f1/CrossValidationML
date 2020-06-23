from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# I det første stykke kode her, har vi et datasæt kaldet dataset
# Vi sætter vores KFold kf til, at splitte 2 gange, og fortæller den ikke, at blande
# Til sidst udskriver vi hvad kf nåede fremtil
# Vi har to split, hvor den første del er trænings data, og den sidste er test data
dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
kf = KFold(n_splits=2, shuffle=False)
for train, test in kf.split(dataset):
    print("%s %s" % (train, test))

# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([1, 2, 3, 4])
# kf = KFold(n_splits=2)

# for train_index, test_index in kf.split(X):
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#    print("%s %s | %s %s" % (X_train, X_test, y_train, y_test))

# Nu vil jeg se på hvordan man kan bruge cross-validation til, at vælge den rigtige model
# Loader iris datasæt
iris = load_iris()
X = iris.data
y = iris.target

print('')
knn = KNeighborsClassifier()

# Tester det optimale antal n_neighbors for knn
for i in range(20):
    knn.n_neighbors = i+1
    print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

# Opstiller de to modeller vi vil stille op mod hinanden
knn.n_neighbors = 20
logreg = LogisticRegression()

# Bruger cross_val_score til at få de to modeller nøjagtigheds scoring
# cv=10 står for, hvor mange fold vi vil have. I dette tilfælde 10
# scoring='accuracy' er hvilken evaluation metric vi har valgt
# Vi bruge mean() tilsidst for, at vi får svaret med det samme. Uden det skulle vi selv beregne gennemsnittet
# Kig nederst for, at se hvad jeg mener.
# Ud fra de svar vi kan vi, så vælge den model der klarede sig bedst.
print('')
print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())

# scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
# print(scores)
# print(scores.mean())
