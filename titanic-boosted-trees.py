from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output
from matplotlib import pyplot as plt

# Seems that the _FeatureColumn APIs are being depreciated, maybe find how to update these
# seems that there are many depreciated or soon-to-be depreciated libraries here.
# Maybe should consider finding a solution to these

#load dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

tf.random.set_seed(123)

# print various info about the dataset
print("\n\n")
#print(dftrain.head(), "\n")
#print(dftrain.describe(), "\n")
#print(dftrain.shape[0], dfeval.shape[0], "\n")

# show graph of passnger ages
#dftrain.age.hist(bins=20)
#plt.show()

# show graph of passenger sex
#dftrain.sex.value_counts().plot(kind='barh')
#plt.show()

# Show graph of passenger class
#dftrain['class'].value_counts().plot(kind='barh')
#plt.show()

# show graph of passenger origin locations
#dftrain['embark_town'].value_counts().plot(kind='barh')
#plt.show()

# Show graph of survival rate by sex
#pd.concat([dftrain, y_train, axis=1).groupby('sex').survived/mean().plot(kind='barh').set_xlabel('% survive')
#plt.show

# Transform CATEGORICAL COLUMNS into one-hot-encoded columns
fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))
#

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # Need to one-hot encode categorical features
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

example = dict(dftrain.head(1))
class_fc = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))

# Print one-hot encoded values
#print('Feature value: "{}"'.format(example['class'].iloc[0]))
#print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())
#print(tf.keras.layers.DenseFeatures(feature_columns)(example).numpy())

# Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    #
    return input_fn
#

#Training and eval input functions
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

linear_est = tf.estimator.LinearClassifier(feature_columns)

# Train model.
linear_est.train(train_input_fn, max_steps=100)

# Evaluation.
result = linear_est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))

# Singe data fits intop memory, use entire dataset per layer. It will be faster.
# Above one batch is defined as the entire dataset.
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=n_batches)

# The model will stop training once the specified number of trees is built, not based on numer of steps.
est.train(train_input_fn, max_steps=100)

# Eval.
result = est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))

# It seems there's something wrong with this block, results in a malloc_consolidate() error
pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred-dicts])
#

probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()


# Show ROC of results 
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel(' false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0, )
plt.ylim(0. )
plt.show()
