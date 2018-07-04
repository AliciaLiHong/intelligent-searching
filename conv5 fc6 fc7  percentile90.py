
# coding: utf-8

# In[1]:


from numpy import *
import os
from pylab import *  #
import numpy as np
from tensorflow.python.ops import nn_ops  #
from tensorflow.python.ops import math_ops #
import matplotlib.pyplot as plt  #
import matplotlib.cbook as cbook #
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg #
from scipy.ndimage import filters
import urllib
from numpy import random
import skimage            #
import imageio            #
import tensorflow as tf  #


# In[2]:


#from caffe_classes import class_names

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]


# In[3]:


net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()


# In[4]:


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        #tf.split(value, num_or_size_splits, axis) -- new version
        ## tf.split(axis, num_or_size_splits, value) -- old version
        
        #input_groups = tf.split(3, group, input)
        #kernel_groups = tf.split(3, group, kernel)
        
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat( output_groups, 3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



x = tf.placeholder(tf.float32, (None,) + xdim)


# In[5]:


k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)

conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])

#fc6 before relu
fc6_before_relu = nn_ops.bias_add( math_ops.matmul(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W)   ,fc6b)



fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

fc7_xw_plus_b = nn_ops.bias_add(math_ops.matmul(fc6, fc7W), fc7b)


fc7_after_relu = nn_ops.relu(fc7_xw_plus_b)


#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(net_data["fc8"][0])
fc8b = tf.Variable(net_data["fc8"][1])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc8)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

t = time.time()


# In[6]:


import PIL
from PIL import Image
import skimage
image_list = []
raw_image_list = []
im_fc7 = []
n_item = 0

for item in os.listdir('tiny-imagenet-200/train/'):
    if item == '.DS_Store':
        continue
    '''if n_item == 200:   # number of classes: 200
        break         ###################
    n_item +=1'''
    for item_2 in os.listdir('tiny-imagenet-200/train/' + item):
        
        if item_2 == 'images':
            n_image = 0   ##################
            for item_3 in os.listdir('tiny-imagenet-200/train/' + item + '/images'):
                try:
                    '''if n_image == 15:  #number of images: 500
                        break           ##############
                    n_image += 1        ##############'''
                    image_name = 'tiny-imagenet-200/train/'+item+'/images/'+item_3
                    #print(image_name)
                
                    im = imageio.imread(image_name)
                    im = imresize(im, (227,227,3))
                    
                    im = im - [0,0,0]
                    #plt.imshow(im)
                    raw_image_list.append(im)
 
                    im = (im).astype(float32)
                    im = im - mean(im)
                    im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
                    image_list.append(im)
                except:
                    pass


# In[7]:


im_fc7 = sess.run(fc7_xw_plus_b, feed_dict = {x:image_list})
im_fc7 = np.array(im_fc7)

im_fc6 = sess.run(fc6_before_relu, feed_dict = {x:image_list})
im_fc6 = np.array(im_fc6)

im_conv5 = sess.run(conv5_in, feed_dict = {x:image_list})
im_conv5 = np.array(im_conv5)


# In[8]:


dataset = 'imagenet'

np.save('data/'+ dataset +'_raw_image_list.npy', raw_image_list)
raw_image_list = np.array(raw_image_list)
print(raw_image_list.shape)

np.save('data/' + dataset + '_image_list.npy', image_list)
image_list = np.array(image_list)
print(image_list.shape)


# In[9]:


np.save('data/' + dataset + '_fc7.npy', im_fc7)
print(im_fc7.shape)

np.save('data/' + dataset + '_fc6.npy', im_fc6)
print(im_fc6.shape)

np.save('data/' + dataset + '_conv5.npy', im_conv5)
print(im_conv5.shape)


# In[10]:


'''get threshold using 90% percentile'''
percentile_fc7 = np.percentile(im_fc7, 90, axis = 0)
percentile_fc6 = np.percentile(im_fc6, 90, axis = 0)
percentile_conv5 = np.percentile(im_conv5, 90, axis = 0)


# In[11]:


np.save('data/' +  dataset +'_percentile_fc7.npy', percentile_fc7)
np.save('data/' +  dataset +'_percentile_fc6.npy', percentile_fc6)
np.save('data/' +  dataset +'_percentile_conv5.npy', percentile_conv5)

percentile_fc7 = np.array(percentile_fc7)
print(percentile_fc7.shape)
percentile_fc6 = np.array(percentile_fc6)
print(percentile_fc6.shape)
percentile_conv5 = np.array(percentile_conv5)
print(percentile_conv5.shape)

