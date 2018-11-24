from __future__ import print_function

import tensorflow as tf
import math
import numpy as np
from matplotlib import pyplot as plt
from modelv import myNeuralNet

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
nn1.addFinalLayer()
nn1.setup_training(0.001,"mnist")
nn1.setup_metrics("mnist")

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)
train_lossarr = []
val_accarr =[]
val_lossarr = []
test_out,train_lossarr,val_lossarr,val_accarr = nn1.train(sess,max_epochs,batch_size,train_size,mnist_train_inp,mnist_train_op,mnist_val_in,mnist_val_op,mnist_test_in,train_lossarr = train_lossarr,val_lossarr=val_lossarr,val_accarr=val_accarr,print_step = 50)
np.save('mnist_out.npy',test_out)
np.save('tl1l500n1e3lr10e100bs.npy',train_lossarr)
np.save('vl1l500n1e3lr10e100bs.npy',val_lossarr)
np.save('va1l500n1e3lr10e100bs.npy',val_accarr)
#print (train_lossarr)
#print (val_lossarr)
#print (val_accarr)
sess.close() # fill in other arguments as you modify the train(self, sess, ...) in model.py
	# you will have to pass xtrain, ytrain, etc ... also as arguments so that you can sample batches in train() of model.py

# write code here to store test_pred in relevant file
#