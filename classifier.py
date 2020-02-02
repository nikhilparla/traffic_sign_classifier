# Load pickled data
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import pickle
import numpy as np
import tensorflow as tf
import os
import cv2
from scipy import ndimage, misc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# TODO: Fill this in based on where you saved the training and testing data

training_file = "./traffic-signs-data/train.p"
validation_file = "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = np.shape(X_train[0])
print(image_shape)

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
X_train_normalized = (X_train - np.float32(128))/np.float32(128)

X_train, y_train = shuffle(X_train_normalized, y_train)

EPOCHS = 100
BATCH_SIZE = 32

# Architecture here.
### Feel free to use as many code cells as needed.

def rotate():
	print("**************Calling rotate function")
	global X_train
	global y_train
	imgs = []
	length = int(len(X_train)/5)
	print("************** shape = " ,X_train.shape)
	print("************** y shape = " ,y_train.shape)
	for i in range(length):
		img_30 = ndimage.rotate(X_train[i],30,reshape=False)	# the new pic doesnt reshape to fit the boundaries
		imgs.append(img_30)
		y_train = np.append(y_train,y_train[i])
	print("*************** 30 deg rotation finished")
	npa = np.asarray(imgs)
	X_train = np.concatenate((X_train,npa), axis=0)
	print("************** shape 2 = " ,X_train.shape)
	print("************** y shape 2= " ,y_train.shape)
	
	X_train, y_train = shuffle(X_train, y_train)
	
	imgs = []
	for i in range(length):
		img_n30 = ndimage.rotate(X_train[i],-30,reshape=False)	# the new pic doesnt reshape to fit the boundaries
		imgs.append(img_30)
		y_train = np.append(y_train,y_train[i])
	npa = np.asarray(imgs)
	X_train = np.concatenate((X_train,npa), axis=0)

	print("**************** -30 deg rotation finished")
		
rotate()
X_train, y_train = shuffle(X_train, y_train)

def sharpen():
	print("**************Calling rotate function")
	global X_train
	global y_train
	imgs = []
	length = int(len(X_train)/5)
	alpha = 30
	
	for i in range(length):
		blurred_f  = ndimage.gaussian_filter(X_train[i], 3)
		filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
		img_sharp = blurred_f + alpha * (blurred_f - filter_blurred_f)
		imgs.append(img_sharp)
		y_train = np.append(y_train,y_train[i])
	print("*************** shapening finished")
	npa = np.asarray(imgs)
	X_train = np.concatenate((X_train,npa), axis=0)

sharpen()
	

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def call_brightness():	
	global X_train
	global y_train
	for i in range(10000, 20000):
		if(i%100 == 0):
			print("completed ",(i/100 - 100 + 1), " percent" )
		frame = increase_brightness(X_train[i],20)
		X_train = np.append(X_train,frame)
		y_train = np.append(y_train,y_train[i])

#call_brightness()

n_train = len(X_train)
print("****************************************")
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print("****************************************")

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    dropout = 0.50  # Dropout, probability to keep units
    
    # Layer 1: Convolution. Input = 32x32x3 Output = 28x28x6
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5,5,3,6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1,1,1,1], padding='SAME') + conv1_b
    
    # Activation
    conv1 = tf.nn.relu(conv1)

    # adding an extra layer
    conv5_w = tf.Variable(tf.truncated_normal(shape=(5,5,6,16), mean = mu, stddev = sigma))
    conv5_b = tf.Variable(tf.zeros(16))
    conv5 = tf.nn.conv2d(conv1, conv5_w, strides=[1,1,1,1], padding='VALID') + conv5_b
        
    # Ac5ivation
    conv5 = tf.nn.relu(conv5)
    
    # Pooling - 28x28x6 to 14x14x6
    conv5_pool = tf.nn.max_pool(conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
        
    # Layer 2: Convolution.  Output = 10x10x6
    conv2_w = tf.Variable(tf.truncated_normal(shape=(5,5,16,16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv5_pool, conv2_w, strides=[1,1,1,1], padding='SAME') + conv2_b
    
    # Activation
    conv2 = tf.nn.relu(conv2)
    
    conv10_w = tf.Variable(tf.truncated_normal(shape=(5,5,16,16), mean = mu, stddev = sigma))
    conv10_b = tf.Variable(tf.zeros(16))
    conv10 = tf.nn.conv2d(conv2, conv10_w, strides=[1,1,1,1], padding='VALID') + conv10_b
    
    # Activation
    conv10 = tf.nn.relu(conv10)
    
    
    # Pooling - 10x10x16 to 5x5x16
    conv2 = tf.nn.max_pool(conv10, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    # Flatten - 5x5x16, Output = 400
    fc0 = flatten(conv2)
    
    
    #Layer 3 - Fully connected . Input = 400, output = 250
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400,250), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(250))
    fc1 = tf.matmul(fc0, fc1_w) + fc1_b
    
    # Activation
    fc1 = tf.nn.relu(fc1)
    
    fc7_w = tf.Variable(tf.truncated_normal(shape=(250,120), mean=mu, stddev=sigma))
    fc7_b = tf.Variable(tf.zeros(120))
    fc7 = tf.matmul(fc1, fc7_w) + fc7_b
    
    # Activation
    fc7 = tf.nn.relu(fc7)
    
    
    #Layer 4 - Fully connected . Input = 120, output = 84
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120,84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc7, fc2_w) + fc2_b
    
    # Activation
    fc2 = tf.nn.relu(fc2)
    
    #Layer 5 - Fully connected . Input = 84, output = 43
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84,43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    
    
    return logits

# placeholder 
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# Training Pipeline

rate = 0.0005

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.compat.v1.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
