from __future__ import print_function

import tensorflow as tf
import math
import numpy as np

from model import myNeuralNet

# x denotes features, y denotes labels
mnist_train_inp = np.load('./data/mnist/xtrain.npy')
mnist_train_op = np.load('./data/mnist/ytrain.npy')
mnist_val_in = np.load('./data/mnist/xval.npy')
mnist_val_op = np.load('./data/mnist/yval.npy')
mnist_test_in = np.load('./data/mnist/xtest.npy')
max_epochs = 10
batch_size = 100
train_size = len(mnist_train_inp)
#input_data = [[1,1,1],[1,-1,1],[1,2,3]]
nn1 = myNeuralNet(784,10)
nn1.addHiddenLayer(500)
nn1.addHiddenLayer(500)
nn1.addFinalLayer()
nn1.setup_training(0.001)
nn1.setup_metrics()

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

test_out = nn1.train(sess,max_epochs,batch_size,train_size,mnist_train_inp,mnist_train_op,mnist_val_in,mnist_val_op,mnist_test_in,50)
np.save('mnist_out.npy',test_out)
sess.close() # fill in other arguments as you modify the train(self, sess, ...) in model.py
	# you will have to pass xtrain, ytrain, etc ... also as arguments so that you can sample batches in train() of model.py

# write code here to store test_pred in relevant file
	