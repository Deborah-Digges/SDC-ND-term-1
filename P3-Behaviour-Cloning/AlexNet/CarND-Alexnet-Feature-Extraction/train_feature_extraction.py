import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np
import math


# TODO: Load traffic signs data.
with open('model/train.p', mode='rb') as f:
    train_data = pickle.load(f)

X_train = train_data['features']
y_train = train_data['labels']
n_classes = len(set(y_train))

# TODO: Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.75, stratify=y_train)

def encode_labels(labels):
    return (np.arange(n_classes) == labels[:,None]).astype(np.float)

y_train = encode_labels(y_train)
y_val = encode_labels(y_val)

NUM_EPOCHS = 20
BATCH_SIZE = 128
NUM_ITERATIONS_PER_EPOCH = math.ceil(X_train.shape[0]/BATCH_SIZE)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

y_true = tf.placeholder(tf.float32, shape=[None, n_classes])

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
fc_8_shape = (fc7.get_shape().as_list()[-1], n_classes)
fc_8_weights = tf.Variable(tf.truncated_normal(shape=fc_8_shape, stddev=1e-2))
fc_8_biases = tf.Variable(tf.zeros(n_classes))

fc_8_logits = tf.nn.xw_plus_b(fc7, fc_8_weights, fc_8_biases)
probs = tf.nn.softmax(fc_8_logits)


# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_8_logits, labels=y_true))

optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(cost)

def accuracy(predictions, true_values):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(true_values, 1))/predictions.shape[0]

# TODO: Train and evaluate the feature extraction model.

with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    for epoch in range(NUM_EPOCHS):

        for idx in range(0, X_train.shape[0], BATCH_SIZE):
            print("Epoch %d, Iteration %d" %(epoch, idx))
            X_batch = X_train[idx: idx + BATCH_SIZE]
            y_batch = y_val[idx: idx + BATCH_SIZE]

            feed_dict = {x: X_batch, y_true: y_batch}

            session.run([optimizer], feed_dict=feed_dict)

        training_cost, training_pred = session.run([cost, probs], feed_dict={x: X_train, y_true: y_train})
        print("Training Accuracy: %.3f, Cost: %.3f for epoch %d" %(accuracy(training_pred, y_train), training_cost, epoch))

    val_cost, val_pred = session.run([cost, probs], feed_dict={x: X_val, y_true: y_val})
    print("Training Accuracy: %.3f, Cost: %.3f" %(accuracy(val_pred, y_val), val_cost))



