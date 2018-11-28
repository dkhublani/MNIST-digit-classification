# Digit-_recognizer_T4_C2
Digit Recognizer - Team 4 - Challenge 2

## Team Contact
* Drupad 979-739-6540
* Mason Rumuly 682-238-2631
* Samyuktha 979-422-2007


## Meetings
* Saturday September 9, meet in Evans Annex 203 at 6:30PM-9PM

## Files
digit recognizer.py: implements single-classifier training and classification
multitrain recognizer.py: implements n-classifier training and classification

## Breakdown
Training over full 60k samples takes prohibitive amount of time and becomes less likely to converge. Partition training set into n equal parts and train n classifiers (nodes), one on each part. For prediction, aggregate in the one of the following ways:
**Mode**: Take each classifiers decision separately. Whichever class is predicted by the most classifiers is considered the prediction.
**Confidence**: This is a fuzzy aggregation scheme. For each prediction, take the sum of log-probabilities of each class accross classifiers. Choose the class with the greatest sum as the prediction.

## Test Result Notes
**3000 training samples and newton-cg solver:**
single-node: 120 seconds to train, 87.11% accuracy 
3-node-mode: 9 seconds to train, 88.17% accuracy
3-node-confidence: 12 seconds train, 89.2% accuracy
Conclusion: multinode increases training speed while preserving accuracy

**60000 training samples and newton-cg solver:**
60-node-mode trained in 212 seconds to 91.4% accuracy
30-node-mode trained in 528 seconds to 92.1% accuracy
20-node-mode trained in 2926 seconds to 92.2% accuracy

60-node-confidence trained in 222 seconds to 91.3% accuracy
30-node-confidence trained in 562 seconds to 91.9% accuracy
20-node-confidence trained in 2641 seconds to 92.3% accuracy

60-node-batch-sum trained in 174 seconds to 91.3% accuracy
60-node-batch-sum trained in 461 seconds to 91.9% accuracy
20-node-batch-sum trained in 4637 seconds to 92.3% accuracy

**60000 training samples and sag solver:**
10-node-batch-sum trained in 8182 seconds to 92.5% accuracy

The sum-confidence 'fuzzy' aggregation does not improve on the modal aggregation; the two are statistically similar.
