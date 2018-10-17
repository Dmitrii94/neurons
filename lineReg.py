import urllib
from urllib import request
import numpy as np
from numpy.linalg import inv

f = urllib.request.urlopen('https://stepic.org/media/attachments/lesson/16462/boston_houses.csv')
bh = np.loadtxt(f, delimiter=',', skiprows=1)
X = bh.copy()
Y = bh[:,0]
X[:,0] = 1

b = ((inv((X.T).dot(X))).dot(X.T)).dot(Y)
print(" ".join(map(str,b)));
# print(b);

# data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with