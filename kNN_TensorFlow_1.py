# ------------------------------------------------------------------
# Filename:    kNN_TensorFlow_1.py
# ------------------------------------------------------------------
# File description:
# Python and TensorFlow image classification using the MNIST dataset.
# ------------------------------------------------------------------

# ------------------------------------------------------
# Modules to import
# ------------------------------------------------------

import tensorflow as tf
import time
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------
# Global variables
# ------------------------------------------------------

k_value_tf = tf.constant(3)


# ------------------------------------------------------
# def create_data_points()
# ------------------------------------------------------
# Create the data for two clusters (cluster 0 and cluster 1)
# Data points in cluster 0 belong to class 0 and data points in
# cluster 1 belong to class 1.
# ------------------------------------------------------
# x is the data point in the cluster and class_value is the class number
# Cluster : a cluster of data point values.
# Class   : the label of the class that the data point belongs to.
# ------------------------------------------------------

def create_data_points():

    print('-- Creating the data points')

    # Cluster 0 data points (x0) / Class 0 label (class_value0 = 0)
    num_points_cluster0 = 100
    mu0 = [-0.5, 5]
    covar0 = [[1.5, 0], [0, 1]]
    x0 = np.random.multivariate_normal(mu0, covar0, num_points_cluster0)
    class_value0 = np.zeros(num_points_cluster0)

    # Cluster 1 data points (x1) / Class 1 label (class_value1= 1)
    num_points_cluster1 = 100
    mu1 = [0.5, 0.75]
    covar1 = [[2.5, 1.5], [1.5, 2.5]]
    x1 = np.random.multivariate_normal(mu1, covar1, num_points_cluster1)
    class_value1 = np.ones(num_points_cluster1)

    print('x0              -> %s' % str(x0))
    print('class_value0     -> %s' % str(class_value0))
    print('x1              -> %s' % str(x1))
    print('class_value1    -> %s' % str(class_value1))

    return x0, class_value0, x1, class_value1


# ------------------------------------------------------
# def create_test_point_to_classify()
# ------------------------------------------------------

def create_test_point_to_classify():

    print('-- Creating a test point to classify')

    data_point = np.array([((np.random.random_sample() * 10) - 5), ((np.random.random_sample() * 10) - 3)])

    data_point_tf = tf.constant(data_point)

    return data_point, data_point_tf


# -------------------------------------------------------------------
# get_label(preds)
# -------------------------------------------------------------------

def get_label(preds):

    print('-- Obtaining the class label')

    counts = tf.math.bincount(tf.dtypes.cast(preds, tf.int32))
    arg_max_count = tf.argmax(counts)

    print('preds       -> %s' % str(preds))
    print('counts      -> %s' % str(counts))
    print('arg_max_count -> %s' % str(arg_max_count))

    return arg_max_count


# -------------------------------------------------------------------
# def predict_class(xt, ct, dt, kt)
# -------------------------------------------------------------------

def predict_class(xt, ct, dt, kt):

    print('-- Predicting the class membership')

    neg_one = tf.constant(-1.0, dtype=tf.float64)
    distance = tf.reduce_sum(tf.abs(tf.subtract(xt, dt)), 1)

    print(neg_one)
    print(distance)

    neg_distance = tf.math.scalar_mul(neg_one, distance)
    # val, val_index = tf.nn.top_k(neg_distance, kt)
    val, val_index = tf.math.top_k(neg_distance, kt)
    cp = tf.gather(ct, val_index)

    print('neg_one      -> %s' % str(neg_one))
    print('distance     -> %s' % str(distance))
    print('neg_distance -> %s' % str(neg_distance))
    print('val          -> %s' % str(val))
    print('val_index    -> %s' % str(val_index))
    print('cp           -> %s' % str(cp))

    return cp


# -------------------------------------------------------------------
# def plot_results(x0, x1, data_point, class_value)
# -------------------------------------------------------------------

def plot_results(x0, x1, data_point, class_value):

    print('-- Plotting the results')

    plt.style.use('default')

    plt.plot(x0[:, 0], x0[:, 1], 'ro', label='class 0')
    plt.plot(x1[:, 0], x1[:, 1], 'bo', label='class 1')
    plt.plot(data_point[0], data_point[1], 'g', marker='D', markersize=10, label='Test data point')
    plt.legend(loc='best')
    plt.grid()
    plt.title('Simple data point classification: Prediction is class %s' % class_value)
    plt.xlabel('Data x-value')
    plt.ylabel('Data y-value')

    plt.show()


# ------------------------------------------------------
# def main()
# ------------------------------------------------------

def main():

    # ------------------------------------------------------
    # -- Start of script run actions
    # ------------------------------------------------------

    print('----------------------------------------------------')
    print('-- Start script run ' + str(time.strftime('%c')))
    print('----------------------------------------------------\n')
    print('-- Python version     : ' + str(sys.version))
    print('-- TensorFlow version : ' + str(tf.__version__))
    print('-- Matplotlib version : ' + str(mpl.__version__))

    # ------------------------------------------------------
    # -- Main script run actions
    # ------------------------------------------------------

    # ------------------------------------------------------
    # 1. Create the data points in each cluster (x0, class_value0, x1, class_value1)
    # 2. Create data point to classify (data_point, data_point_tf)
    # 3. Combine all cluster values into combined lists (x & class_value)
    # 4. Convert (x & class_value) values to TensorFlow constants (x_tf & class_value_tf)
    # ------------------------------------------------------

    (x0, class_value0, x1, class_value1) = create_data_points()
    (data_point, data_point_tf) = create_test_point_to_classify()

    x = np.vstack((x0, x1))
    class_value = np.hstack((class_value0, class_value1))

    x_tf = tf.constant(x)
    class_value_tf = tf.constant(class_value)

    print('x_tf -> %s' % str(x_tf))
    print('class_value_tf   -> %s' % str(class_value_tf))
    print('x                -> %s' % str(x))
    print('class_value      -> %s' % str(class_value))

    # ------------------------------------------------------
    # Run TensorFlow to predict the classification of data point and
    # print the predicted class using nearest 'k_value' data points.
    # ------------------------------------------------------

    pred = predict_class(x_tf, class_value_tf, data_point_tf, k_value_tf)
    class_value_index = pred
    class_value = get_label(class_value_index)

    print(pred)
    print(class_value_index)
    print(class_value)

    print('\n-----------------------------------------------------------')
    print('-- Prediction: data point %s is in class %s' % (str(data_point), class_value))
    print('-----------------------------------------------------------\n')

    # ------------------------------------------------------
    # Plot the data points
    # ------------------------------------------------------

    plot_results(x0, x1, data_point, class_value)

    # ------------------------------------------------------
    # -- End of script run actions
    # ------------------------------------------------------

    print('----------------------------------------------------')
    print('-- End script run ' + str(time.strftime('%c')))
    print('----------------------------------------------------\n')


# ------------------------------------------------------
# Run only if source file is run as the main script
# ------------------------------------------------------

if __name__ == '__main__':

    main()

# ------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------
