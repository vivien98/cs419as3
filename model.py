import tensorflow as tf
import numpy as np
import math
# include any other imports that you want

'''
This file contains a class for you to implement your neural net.
Basic function skeleton is given, and some comments to guide you are also there.

You will find it convenient to look at the tensorflow API to understand what functions to use.
'''

'''
Implement the respective functions in this class
You might also make separate classes for separate tasks , or for separate kinds of networks (normal feed-forward / CNNs)
'''
class myNeuralNet:
	# you can add/modify arguments of *ALL* functions
	# you might also add new functions, but *NOT* remove these ones
	def __init__(self, dim_input_data, dim_output_data): # you can add/modify arguments of this function 
		# Using such 'self-ization', you can access these members in later functions of the class
		# You can do such 'self-ization' on tensors also, there is no change
		self.dim_input_data = dim_input_data
		self.dim_output_data = dim_output_data
		self.num_perc_arr = [dim_input_data,dim_output_data]

		self.inp = tf.placeholder(tf.float32,[None,dim_input_data])
		self.oput = tf.placeholder(tf.float32,[None,dim_output_data])
		first_layer_weight = tf.Variable(tf.random_normal([dim_input_data,dim_output_data]))
		first_layer_bias = tf.Variable(tf.zeros([1,dim_output_data]))

		self.layer_weight_list = [first_layer_weight]
		self.layer_bias_list = [first_layer_bias]
		# Create placeholders for input : data as well as labels
		# You might want to initialising some container to store all the layers of the network

	def addHiddenLayer(self, layer_dim, activation_fn=None, regularizer_fn=None):
		# Add a layer to the network of layer_dim
		# It might be a good idea to append the new layer to the container of layers that you initialized before
		self.num_perc_arr[-1] = (layer_dim)
		self.num_perc_arr.append(self.dim_output_data)

		del self.layer_weight_list[-1]
		self.layer_weight_list.append(tf.Variable(tf.random_normal([self.num_perc_arr[-3],self.num_perc_arr[-2]])))
		self.layer_weight_list.append(tf.Variable(tf.random_normal([self.num_perc_arr[-2],self.num_perc_arr[-1]])))
		self.layer_bias_list[-1] = tf.Variable(tf.random_normal([1,layer_dim]))
		first_layer_bias = tf.Variable(tf.random_normal([1,self.dim_output_data]))
		self.layer_bias_list.append(first_layer_bias)

		pass

	def addFinalLayer(self, activation_fn=None, regularizer_fn=None):
		# We don't take layer_dim here, since the dimensionality of final layer is
		# already stored in self.dim_output_data
		fwd_pass = self.inp
		
		for i in range(len(self.layer_weight_list)-1):
			fwd_pass = tf.add(tf.matmul(fwd_pass,self.layer_weight_list[i]),self.layer_bias_list[i])
			fwd_pass = tf.nn.relu(fwd_pass)
		fwd_pass = tf.add(tf.matmul(fwd_pass,self.layer_weight_list[-1]),self.layer_bias_list[-1])
		self.mlp_out = fwd_pass

		# Create the output of the final layer as logits
		# You might also like to apply the final activation function (softmax / sigmoid) to get the predicted labels
		
	
	def setup_training(self, learn_rate, loss_type):
		# Define loss, you might want to store it as self.loss
		# Define the train step as self.train_step = ..., use an optimizer from tf.train and call minimize(self.loss)
		if(loss_type == "mnist"):
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.oput,logits = self.mlp_out))
		elif(loss_type == "speech"):
			self.loss = tf.reduce_mean(tf.nn.l2_loss(self.oput-tf.sign(self.mlp_out)))
		self.train_step = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)
		pass

	def setup_metrics(self):
		# Use the predicted labels and compare them with the input labels(placeholder defined in __init__)
		# to calculate accuracy, and store it as self.accuracy
		compare = tf.equal(tf.argmax(self.mlp_out,1),tf.argmax(self.oput,1))
		self.accuracy = tf.reduce_mean(tf.cast(compare,tf.float32))
		pass
	
	# you will need to add other arguments to this function as given below
	def train(self, sess, max_epochs, batch_size, train_size,train_inp,train_op,val_inp,val_op,test_inp, print_step = 100): # valid_size, test_size, etc
		# Write your training part here
		# sess is a tensorflow session, used to run the computation graph
		# note that all the functions uptil now were just constructing the computation graph
		
		# one 'epoch' represents that the network has seen the entire dataset once - it is just standard terminology
		steps_per_epoch = int(train_size/batch_size)
		train_sub_in = np.array_split(train_inp,steps_per_epoch)
		train_sub_out = np.array_split(train_op,steps_per_epoch)
		max_steps = max_epochs * steps_per_epoch
		for step in range(max_steps):
			# read a batch of data from the training data
			# now run the train_step, self.loss on this batch of training data. something like :
			in_batch = train_sub_in[step%steps_per_epoch]
			out_batch = train_sub_out[step%steps_per_epoch] 
			_, train_loss = sess.run(	[self.train_step, self.loss], feed_dict={self.inp:in_batch , self.oput: out_batch})
			if (step % print_step) == 0:
				# read the validation dataset and report loss, accuracy on it by running

				val_acc, val_loss, val_pred = sess.run([self.accuracy, self.loss, tf.sign(self.mlp_out)], feed_dict={self.inp:val_inp , self.oput:val_op })
				print(val_pred)
				#print(val_loss)
				#print(val_acc)
				# remember that the above will give you val_acc, val_loss as numpy values and not tensors
				# store these train_loss and validation_loss in lists/arrays, write code to plot them vs steps
			# Above curves are *REALLY* important, they give deep insights on what's going on
		# -- for loop ends --
		# Now once training is done, run predictions on the test set

		self.test_pred = tf.argmax(self.mlp_out,1)
		test_predictions = sess.run([self.test_pred], feed_dict={self.inp:test_inp })
		return test_predictions
		# This is because we will ask you to submit test_predictions, and some marks will be based on how your net performs on these unseen instances (test set)
		'''
		We have done everything in train(), but
		you might want to create another function named eval(),
		which calculates the predictions on test instances ...
	
		'''


'''
pr1=sess.run([nn1.mlp_out],feed_dict={nn1.inp:input_data})
pr3 = sess.run(nn1.layer_weight_list)
pr2 = sess.run(nn1.layer_bias_list)
print(pr3)
print(pr2)
print(pr1)
'''

'''
	NOTE:
	you might find it convenient to make 3 different train functions corresponding to the three different tasks,
	and call the relevant one from each train_*.py
	The reason for this is that the arguments to the train() are different across the tasks
'''
'''
	Example, for the speech part, the train() would look something like :
	(NOTE: this is only a rough structure, we don't claim that this is exactly what you have to do.)
	
	train(self, sess, batch_size, train_size, max_epochs, train_signal, train_lbls, valid_signal, valid_lbls, test_signal):
		steps_per_epoch = math.ceil(train_size/batch_size)
		max_steps = max_epochs*steps_per_epoch
		print(max_steps)
		for step in range(max_steps):
			# select batch_size elements randomly from training data
			sampled_indices = random.sample(range(train_size), batch_size)
			trn_signal = train_signal[sampled_indices]
			trn_labels = train_lbls[sampled_indices]
			if (step % steps_per_epoch) == 0:
				val_loss, val_acc = sess.run([self.loss, self.accuracy], feed_dict={input_data: valid_signal, input_labels: valid_lbls})
				print(step, val_acc, val_loss)
			sess.run(self.train_step, feed_dict={input_data: trn_signal, input_labels: trn_labels})
		test_prediction = sess.run([self.predictions], feed_dict={input_data: test_signal})
		return test_prediction
'''