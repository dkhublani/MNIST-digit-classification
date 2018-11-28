
from mnist import MNIST
from time import time
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import numpy as np

# settings
result_file_prefix = 'results/knn_'

# Load mnist datasets
start = time()
mndata = MNIST('mnist-data')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
end = time()
print("load completed in {} seconds".format(int(end - start)))
start = time()
# truncate set
#train_images = train_images[:100]
#train_labels = train_labels[:100]

classifier = KNeighborsClassifier(n_neighbors = 5, p = 120)
classifier.fit(train_images, train_labels)

# y_pred = classifier.predict(test_images)
# y_score = classifier.score(test_images, list(test_labels))

# print(y_score)

# classify submission set
target = pd.read_csv('submission-data/mnist_test.csv', index_col=0)
result = classifier.predict(target)
df = pd.DataFrame({'Category':result})
df.index.name = 'Id'
df.to_csv(result_file_prefix + "submission.csv", sep=',', encoding='utf-8')
end = time()

print(end - start)





