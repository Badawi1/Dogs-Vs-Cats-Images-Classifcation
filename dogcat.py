import glob
import cv2
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import time


filenames = os.listdir("C:/D Partition/4th year first term\ml/3rd_ass/train")
newfile = filenames[0:3000] + filenames[12500:15500]

categories = []
for filename in newfile:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': newfile,
    'category': categories
})
df = df.sample(frac=1,random_state=0).reset_index(drop=True)

images = []
fds = []
start_time = time.time()
for index,row in df.iterrows():
    name = row['filename']
    img = imread("C:/D Partition/4th year first term\ml/3rd_ass/train/"+name)
    resized_img = resize(img, (128,64))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
    images.append(resized_img)
    fds.append(fd)

print("---extracting features time %s seconds ---" % (time.time() - start_time))

Y = df['category']

X_train, X_test, y_train, y_test = train_test_split(fds, Y, test_size = 0.20,random_state=0)
#rbf_svc = svm.SVC(kernel='rbf', gamma='scale', C=0.1)
#rbf_svc = svm.SVC(kernel='linear', C=0.1)
rbf_svc = svm.SVC(kernel='poly', degree=3, C=0.1)
#rbf_svc = svm.LinearSVC(C=0.01)
'''start_time = time.time()
scores = cross_val_score(rbf_svc,X_train, y_train, scoring='accuracy', cv=10)
print("---cross validation time %s seconds ---" % (time.time() - start_time))
cv_score = abs(scores.mean())
print("cross validation score is "+ str(cv_score))'''
start_time = time.time()
rbf_svc.fit(X_train,y_train)
print("---training time %s seconds ---" % (time.time() - start_time))
predictions = rbf_svc.predict(X_test)
print("Accuracy is ",accuracy_score(y_test,predictions))



