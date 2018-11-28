from mnist import MNIST
from sklearn.linear_model import LogisticRegression
from time import time

# Load mnist datasets
start = time()
mndata = MNIST('mnist-data')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
end = time()
print("load completed in {} seconds".format(int(end - start)))

# truncate set
train_images = train_images[:200]
train_labels = train_labels[:200]
# test_images = test_images[:]
# test_labels = test_labels[:]

# Train logistic regression classifiers
part_A = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000, n_jobs=1)
part_B = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000, n_jobs=1)
total = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000, n_jobs=1)
start = time()
part_A = part_A.fit(train_images[:100], train_labels[:100])
part_B = part_B.fit(train_images[100:], train_labels[100:])
total = total.fit(train_images[:], train_labels[:])
end = time()
print("train completed in {} seconds".format(int(end - start)))

# Check accuracy
accuracy_a = part_A.score(test_images, list(test_labels))
print("part A accuracy on test set is {} percent".format(accuracy_a * 100))
accuracy_b = part_B.score(test_images, list(test_labels))
print("part B accuracy on test set is {} percent".format(accuracy_b * 100))
accuracy_t = total.score(test_images, list(test_labels))
print("total accuracy on test set is {} percent".format(accuracy_t * 100))

# sum matrixes for sets
combined = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000, n_jobs=1)
combined.coef_ = (part_A.coef_ + part_B.coef_)/2
combined.intercept_ = (part_A.intercept_ + part_B.intercept_)/2
combined.classes_ = part_A.classes_
accuracy_c = combined.score(test_images, list(test_labels))
print("summed coefficient accuracy on test set is {} percent".format(accuracy_c * 100))