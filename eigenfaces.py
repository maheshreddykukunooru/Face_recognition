from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy import misc
import os


unwantedList = ['Readme.txt', 'subject01.centerlight', 'subject01.centerlight.gif']
images = os.listdir('./yalefaces/')
images = [image for image in images if image not in unwantedList]
# print len(images)

X=[]
y=[]
for image in images:
	face = misc.imread('./yalefaces/'+image)
	subject = int(image.split('.')[0][-2:])
	f=misc.face(gray=True)
	[width1,height1]=[f.shape[0],f.shape[1]]
	f2=f.reshape(width1*height1)
	X.append(f2)
	y.append(subject)

target_names = []
for i in range(15):
	if i < 9:
		name = "subject0"
	else:
		name = "subject"
	name+=str(i+1)
	target_names.append(name)

n_classes = 15
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

n_components = 70
print len(X_train),len(X_test),len(y_train),len(y_test)

pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

print "pca fit done"
# eigenfaces = pca.components_.reshape((n_components, width1, height1))

# X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test)

# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
# clf = clf.fit(X_train_pca, y_train)

# print(clf.best_estimator_)

# y_pred = clf.predict(X_test_pca)

# print(classification_report(y_test, y_pred, target_names=target_names))
# print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

