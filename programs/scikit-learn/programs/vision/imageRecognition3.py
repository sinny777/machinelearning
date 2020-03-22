import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

# download dataset from http://ufldl.stanford.edu/housenumbers/
# load our dataset
train_data = scipy.io.loadmat('/../data/images/house_numbers/extra_32x32.mat')
# extract the images and labels from the dictionary object
X = train_data['X']
y = train_data['y']

# plt.imshow(X[:,:,:,img_index])
# plt.show()
# print(y[img_index])

X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]).T
y = y.reshape(y.shape[0],)
X, y = shuffle(X, y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_test.shape)
# load the model from disk
filename = 'my_image_classifier.pkl'
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print(result)
