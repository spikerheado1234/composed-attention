import tensorflow as tf
import os
import tensorflow_datasets as tfds

## Load the IMDB dataset. ##
train_data, val_data, test_data = tfds.load(name="imdb_reviews", split=('train[:60%]', 'test[60%:]', 'test'), as_supervised=True)

## Just testing if the Dataset is loaded. Seems OK ##
train_batch, train_labels = next(iter(train_data.batch(10)))

