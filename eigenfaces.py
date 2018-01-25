from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import os

import cv2

X=[]
y=[]
for i in range(1,41):
	images = os.listdir('./att_faces/s'+str(i))
	for image in images:

		img = cv2.imread('./att_faces/s'+str(i)+"/"+image,0)
		height1, width1 = img.shape[:2]
		img_col = np.array(img, dtype='float64').flatten()
		subject = int(i)
		X.append(img_col)
		y.append(subject)

target_names = []
for i in range(1,41):
	name=''
	name+='s'+str(i)
	target_names.append(name)

n_classes = 40
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42)

n_components = 20

pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

eigenfaces = pca.components_.reshape((n_components, height1, width1))

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)

print(clf.best_estimator_)

y_pred = clf.predict(X_test_pca)

print(classification_report(y_test, y_pred, target_names=target_names))

