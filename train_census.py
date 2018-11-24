from __future__ import print_function

import numpy as np
import tensorflow as tf
import numpy as np
import math

from model import myNeuralNet
import pandas as pd

'''
For this task, the inputting code is not fully given to you.
The reason for this is that there are two important data cleaning tasks that you need to handle
1. Handling categorical values
	Hint: Look up one-hot encoding on the internet
2. Handling missing attributes
	Hint: You might want to replace such missing values with a reasonable statistical measure of that attribute

For more information, look at /data/census/description
'''
def get_type(inp,sess):
	print(sess.run(inp))
	a = False
	voc = tf.constant([" Private"," Self-emp-not-inc"," Self-emp-inc"," Federal-gov"," Local-gov"," State-gov"," Without-pay"," Never-worked"])	
	table = tf.contrib.lookup.index_table_from_tensor(voc,default_value = 0)
	encoded = tf.one_hot(table.lookup(inp),8,dtype=tf.int8)
	return encoded

dim_input = 104 #using one-hot encoding
dim_output = 1 # binary class classification can be done using sigmoid

max_epochs = 50
learn_rate = 1e-4
batch_size = 50

def file_size(fname):
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1

train_fname = 'data/census/train_data'
valid_fname = 'data/census/validation_data'
test_fname = 'data/census/test_data'
train_size = file_size(train_fname)
valid_size = file_size(valid_fname)
test_size = file_size(test_fname)

print(train_size)
print(valid_size)
print(test_size)
# Import data
'''
Note:
In this framework, we'll read the files as it is and perform data "cleaning" once we've read the batches.
You might want to go for the other route(like in train_mnist.py) - where the entire data is read in at once
and then you can perform data manipulations on it.

Also note that the last column is -1 in the test dataset.
This is to make sure that the same code works for all the three inputting framework,
but the input labels in test case are garbage (since they are all -1). 
'''
#---------------------------------------------data acquisition-------------------------------------
'''train_dat = pd.read_csv(train_fname,delimiter=",")
for i in range (len(train_dat)):
	for j in range (15):
		a = train_dat.iloc[i,j]
		if(a == " ?"):
			train_dat.iloc[i,j] = train_dat.iloc[0,j]
		elif(a == " <=50K."):
			train_dat.iloc[i,j] = " <=50K"
		elif(a == " >50K."):
			train_dat.iloc[i,j] = " >50K"		
df1 = train_dat.ix[:,["0","2","4","10","11","12"]]
#rint (df1)
#print (train_dat["1"])
df2 = pd.get_dummies(train_dat["1"])
df3 = pd.get_dummies(train_dat["3"])
df5 = pd.get_dummies(train_dat["5"])
df6 = pd.get_dummies(train_dat["6"])
df7 = pd.get_dummies(train_dat["7"])
df8 = pd.get_dummies(train_dat["8"])
df9 = pd.get_dummies(train_dat["9"])
df13 = pd.get_dummies(train_dat["13"])
final_train = pd.concat([df1,df2,df3,df5,df6,df7,df8,df9,df13],axis=1,sort=False)
column_list = (list(final_train))
final_train_labels = train_dat["14"]
train_data = final_train.values
train_labels = final_train_labels.values
#print (train_data.shape)
#print(train_labels)
np.save("./census_train.npy",train_data)
np.save("./census_train_labels.npy",train_labels)

# dataset_valid = tf.contrib.data.make_csv_dataset(valid_fname, valid_size,column_names = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14"]) # all validation instances in a single tensor
# iterator_valid = dataset_valid.make_initializable_iterator()
# valid_batcher = iterator_valid.get_next()
train_dat = pd.read_csv(valid_fname,delimiter=",")
for i in range (len(train_dat)):
	for j in range (15):
		a = train_dat.iloc[i,j]
		if(a == " ?"):
			train_dat.iloc[i,j] = train_dat.iloc[0,j]
		elif(a == " <=50K."):
			train_dat.iloc[i,j] = " <=50K"
		elif(a == " >50K."):
			train_dat.iloc[i,j] = " >50K"		
df1 = train_dat.ix[:,["0","2","4","10","11","12"]]
#rint (df1)
#print (train_dat["1"])
df2 = pd.get_dummies(train_dat["1"])
df3 = pd.get_dummies(train_dat["3"])
df5 = pd.get_dummies(train_dat["5"])
df6 = pd.get_dummies(train_dat["6"])
df7 = pd.get_dummies(train_dat["7"])
df8 = pd.get_dummies(train_dat["8"])
df9 = pd.get_dummies(train_dat["9"])
df13 = pd.get_dummies(train_dat["13"])
final_valid = pd.concat([df1,df2,df3,df5,df6,df7,df8,df9,df13],axis=1,sort=False)
final_valid_labels = train_dat["14"]
#final_valid = final_valid.drop(" Holand-Netherlands",axis = 1)
final_valid = final_valid.loc[:,column_list].fillna(0)
print (list(final_valid))
print (len(list(final_valid)))
print (len(list(final_train)))
valid_data = final_valid.values
valid_labels = final_valid_labels.values
np.save("./census_valid.npy",valid_data)
#print (valid_data.shape)
np.save("./census_valid_labels.npy",valid_labels)
# dataset_test = tf.contrib.data.make_csv_dataset(test_fname, test_size,column_names = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14"]) # all test instances in a single tensor
# iterator_test = dataset_test.make_initializable_iterator()
# test_batcher = iterator_test.get_next()
train_dat = pd.read_csv(test_fname , delimiter="," , header = None,names = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14"])
print (list(train_dat))
for i in range (len(train_dat)):
	for j in range (15):
		a = train_dat.iloc[i,j]
		if(a == " ?"):
			train_dat.iloc[i,j] = train_dat.iloc[0,j]		
df1 = train_dat.ix[:,["0","2","4","10","11","12"]]
#rint (df1)
#print (train_dat["1"])
df2 = pd.get_dummies(train_dat["1"])
df3 = pd.get_dummies(train_dat["3"])
df5 = pd.get_dummies(train_dat["5"])
df6 = pd.get_dummies(train_dat["6"])
df7 = pd.get_dummies(train_dat["7"])
df8 = pd.get_dummies(train_dat["8"])
df9 = pd.get_dummies(train_dat["9"])
df13 = pd.get_dummies(train_dat["13"])
final_test = pd.concat([df1,df2,df3,df5,df6,df7,df8,df9,df13],axis=1,sort=False)
final_test = final_test.loc[:,column_list].fillna(0)
print (len(list(final_test)))
print(list(final_test))	
test_data = final_test.values
np.save("./census_test.npy",test_data)'''
train_data = np.load("census_train.npy")
train_labels = np.load("census_train_labels.npy")
#print (train_labels)
k = []
for i in range(len(train_labels)):
	if(train_labels[i] == " <=50K"):
		k.append(0)
	else:
		k.append(1)
train_labels = k			
valid_data = np.load("census_valid.npy")
valid_labels = np.load("census_valid_labels.npy")
k = []
for i in range(len(valid_labels)):
	if(valid_labels[i] == " <=50K"):
		k.append(0)
	else:
		k.append(1)
valid_labels = k
# print (valid_labels)
# print (train_labels)			
test_data = np.load("census_test.npy")
print (test_data.shape)
print (train_data.shape)
print(len(train_labels))
print (valid_data.shape)
print(len(valid_labels))
#--------------------------------------end of data acquisuition-------------------------------------

# Create Computation Graph
nn_instance = myNeuralNet(dim_input, dim_output)
nn_instance.addHiddenLayer(20, activation_fn=tf.nn.relu)
#nn_instance.addHiddenLayer(200, activation_fn=tf.nn.relu)
# add more hidden layers here by calling addHiddenLayer as much as you want
# a net of depth 3 should be sufficient for most tasks
nn_instance.addFinalLayer()
nn_instance.setup_training(learn_rate,"mnist")
nn_instance.setup_metrics()

#setup variables
max_epochs = 10
batch_size = 100
train_size = len(train_data)

# # Instantiate Session
sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

test_out = nn_instance.train(sess,max_epochs,batch_size,train_size,train_data,train_labels,valid_data,valid_labels,test_data,50)
np.save('census_out.npy',test_out)
sess.close()
# with tf.Session() as sess:
# 	sess.run([iterator_train.initializer, iterator_valid.initializer, iterator_test.initializer])
# 	sess.run(tf.global_variables_initializer())
# 	get_clean_batches(train_batcher,sess)
	#test_pred = nn_instance.train(sess) # add more arguments here
	# you will have to pass the train_batcher, valid_batcher, test_batcher to this for it to batch

	# In this framework, you will have to perform cleaning of data, and for such things you might want to
	# make use of get_clean_batches ...
	# For that, you'll have to import that in model.py

# write code here to store test_pred in relevant file