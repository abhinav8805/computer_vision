import os
import pickle
import zipfile
import gdown
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#data reading
file_id = '12zOiqLWDRUM7CB6s4rDQssheCIiOhoQz'
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, 'data_file.zip', quiet=False)

#unzip and read the dat
with zipfile.ZipFile('data_file.zip', 'r') as zip_ref:
    zip_ref.extractall('data_file')

# prepare data
input_dir = 'data_file/clf-data'

categories = ['empty', 'not_empty'] #enter the 2 options

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train_test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# classifier
classifier = SVC()

#12 models 
parameters = [{'gamma': [0.01, 0.001], 'C': [1, 10]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance score
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))