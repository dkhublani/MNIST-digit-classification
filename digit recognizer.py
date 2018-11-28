from mnist import MNIST
from sklearn.linear_model import LogisticRegression
from time import time
import numpy as np
import pandas as pd

# settings
result_file_prefix = 'results/logreg_'

# Load mnist datasets
start = time()
mndata = MNIST('mnist-data')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
end = time()
print("load completed in {} seconds".format(int(end - start)))

# truncate set
# train_images = train_images[:100]
# train_labels = train_labels[:100]
# test_images = test_images[:]
# test_labels = test_labels[:]

# Train logistic regression classifier
logreg = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=100000, n_jobs=-1)
start = time()
logreg = logreg.fit(train_images, train_labels)
end = time()
print("train completed in {} seconds".format(int(end - start)))

# Check accuracy
accuracy = logreg.score(test_images, list(test_labels))
print("accuracy on test set is {} percent".format(accuracy * 100))

# save coefficients
np.savetxt(result_file_prefix + "coefficients.csv", logreg.coef_, delimiter=",")
np.savetxt(result_file_prefix + "intercepts.csv", logreg.intercept_, delimiter=",")
np.savetxt(result_file_prefix + "classes.csv", logreg.classes_, delimiter=",")
# note conditions
f = open(result_file_prefix + 'execute_notes.txt', 'w')
f.write('accuracy on test set is {} percent'.format( accuracy * 100))
f.close()

# classify submission set
target = pd.read_csv('submission-data/mnist_test.csv', index_col=0)
result = logreg.predict(target)
df = pd.DataFrame({'Category':result})
df.index.name = 'Id'
df.to_csv(result_file_prefix + "submission.csv", sep=',', encoding='utf-8')
