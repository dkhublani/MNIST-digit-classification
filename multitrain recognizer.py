from mnist import MNIST
from sklearn.linear_model import LogisticRegression
from time import time
from scipy.stats import mode
import numpy as np
import pandas as pd

# settings
n_classifiers = 10
result_file_prefix = 'results/logreg_batch_'

# Load mnist datasets
start = time()
mndata = MNIST('mnist-data')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
end = time()
print("load completed in {} seconds".format(int(end - start)))

# truncate set
# train_images = train_images[:600]
# train_labels = train_labels[:600]
# test_images = test_images[:]
# test_labels = test_labels[:]

# make and train disjoint-trained classifiers
start = time()
n_samples = len(train_labels) // n_classifiers
classifiers = [
    LogisticRegression(
        multi_class='multinomial', 
        solver='sag', 
        max_iter=10000, 
        n_jobs=-1
    ).fit(
        train_images[i*n_samples:(i+1)*n_samples], 
        train_labels[i*n_samples:(i+1)*n_samples]
    )
    for i in range(n_classifiers)
]
combined = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=10000, n_jobs=-1)
combined.coef_ = sum([c.coef_ for c in classifiers])
combined.intercept_ = sum([c.intercept_ for c in classifiers])
combined.classes_ = classifiers[0].classes_ # Assumes first has. Easiest for now
end = time()
print("train completed in {} seconds".format(int(end - start)))

# Check accuracy
accuracy = combined.score(test_images, list(test_labels))
print("accuracy on test set is {} percent".format(accuracy * 100))

# save coefficients
np.savetxt(result_file_prefix + "coefficients.csv", combined.coef_, delimiter=",")
np.savetxt(result_file_prefix + "intercepts.csv", combined.intercept_, delimiter=",")
np.savetxt(result_file_prefix + "classes.csv", combined.classes_, delimiter=",")
# note conditions
f = open(result_file_prefix + 'execute_notes.txt', 'w')
f.write('accuracy with {} batches on test set is {} percent'.format(n_classifiers, accuracy * 100))
f.close()

# classify submission set
target = pd.read_csv('submission-data/mnist_test.csv', index_col=0)
result = combined.predict(target)
df = pd.DataFrame({'Category':result})
df.index.name = 'Id'
df.to_csv(result_file_prefix + "submission.csv", sep=',', encoding='utf-8')
