import os
import pickle
from skimage.io import imread
from skimage.transform import resize

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data_directory = '/Users/ASUS/Downloads/ParkingLotDetectorAndCounter-20250125T203719Z-001/ParkingLotDetectorAndCounter/clf-data'
categories = ['empty', 'not_empty']

images_data = []
image_labels = []

for category_index, category in enumerate(categories):
    category_path = os.path.join(data_directory, category)
    for file_name in os.listdir(category_path):
        image_path = os.path.join(category_path, file_name)
        image = imread(image_path)
        resized_image = resize(image, (15, 15))
        images_data.append(resized_image.flatten())
        image_labels.append(category_index)

images_data = np.asarray(images_data)
image_labels = np.asarray(image_labels)

x_train, x_test, y_train, y_test = train_test_split(
    images_data, image_labels, test_size=0.2, shuffle=True, stratify=image_labels
)

svm_classifier = SVC()

hyperparameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(svm_classifier, hyperparameters)

grid_search.fit(x_train, y_train)

best_classifier = grid_search.best_estimator_

predicted_labels = best_classifier.predict(x_test)

accuracy = accuracy_score(predicted_labels, y_test)
print('{}% of samples were correctly classified'.format(str(accuracy * 100)))

pickle.dump(best_classifier, open('./best_svm_model.p', 'wb'))
