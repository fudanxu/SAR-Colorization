# -*- coding: utf-8 -*-
"""
main.py
Code for paper: Code for Paper Q. Song, F. Xu, and Y.Q. Jin, 
"Radar Image Colorization: Converting Single-Polarization to 
Fully Polarimetric Using Deep Neural Networks," IEEE Access.

SAR Image Colorization: Reconstruction full-pol data from single-pol data.
by Qian Song on Feb. 26th 2017
"""


#import scipy.misc
import numpy as np
import tensorflow as tf
import scipy.io as sio
batch_size = 1
input_size = 400
feature_size = 1153
checkpoint_dir = './checkpoint'

matfn = './data/const_array.mat'
data1 = sio.loadmat(matfn)
const_array = data1['const_array']
const_array[1,31] = 0.3
   

def weights(shape,name1):
    weights_dic = load_weights();    
    if name1 in weights_dic:
        if name1=='wconv1_1':
            weights_value = np.mean(weights_dic[name1],axis = 2)
            weights_value.shape = 3,3,1,64
            w = tf.Variable(weights_value,name = name1,trainable = False)
        else:
            w = tf.Variable(weights_dic[name1],name = name1,trainable = False)
    else:
        w = tf.Variable(tf.random_normal(shape = shape,stddev= np.sqrt(2.0/shape[0])),name = name1,trainable = True)
    return w

def bias(shape,name2):    
    weights_dic = load_weights();
    if name2 in weights_dic:
        temp_b = weights_dic[name2]
        temp_b.shape = shape
        b = tf.Variable(temp_b,name = name2,trainable = False)
    else:
        b = tf.Variable(tf.constant(0.0,"float32",shape),name = name2,trainable = True)
    return b
    
def conv2d(input_,w,bia,strides=[1, 1, 1, 1],paddings = "SAME"):
    conv = tf.nn.conv2d(input_,w,strides,padding= paddings)
    return tf.nn.bias_add(conv,bia)
    
def load_weights():
    matfn = './data/vgg16_tf.mat'
    dic = sio.loadmat(matfn)
    return dic
    
def Bilinear(IM):
    output = tf.image.resize_bilinear(IM,size = [input_size,input_size],align_corners=True)
    return output
                     
def VGG16(X):
    #the first layer
    wconv1_1 = weights([3,3,1,64],name1 = 'wconv1_1')    
    bconv1_1 = bias([64],name2 = 'bconv1_1')
    wconv1_2 = weights([3,3,64,64],name1 = 'wconv1_2')
    bconv1_2 = bias([64],name2 = 'bconv1_2')
    
    conv1_1 =  conv2d(X,wconv1_1,bconv1_1)
    conv1_2 =  conv2d(conv1_1,wconv1_2,bconv1_2)
    
    conv1_1 =  tf.nn.relu(conv1_1)
    conv1_2 =  tf.nn.relu(conv1_2)
    conv1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding= "SAME")
    
  #the second layer
    wconv2_1 = weights([3,3,64,128],name1 = 'wconv2_1')
    bconv2_1 = bias([128],name2 = 'bconv2_1')
    wconv2_2 = weights([3,3,128,128],name1 = 'wconv2_2')
    bconv2_2 = bias([128],name2 = 'bconv2_2')

    conv2_1 =  tf.nn.relu(conv2d(conv1,wconv2_1,bconv2_1))
    conv2_2 =  tf.nn.relu(conv2d(conv2_1,wconv2_2,bconv2_2))
    conv2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding= "SAME")
    
   
    wconv3_1 = weights([3,3,128,256],name1 = 'wconv3_1')
    bconv3_1 = bias([256],name2 = 'bconv3_1')
    wconv3_2 = weights([3,3,256,256],name1 = 'wconv3_2')
    bconv3_2 = bias([256],name2 = 'bconv3_2')
    wconv3_3 = weights([3,3,256,256],name1 = 'wconv3_3')
    bconv3_3 = bias([256],name2 = 'bconv3_3')

    conv3_1 =  tf.nn.relu(conv2d(conv2  ,wconv3_1,bconv3_1))
    conv3_2 =  tf.nn.relu(conv2d(conv3_1,wconv3_2,bconv3_2))
    conv3_3 =  tf.nn.relu(conv2d(conv3_2,wconv3_3,bconv3_3))

    print "VGG16 has been loaded!"
    H = tf.concat_v2(axis = 3,values = [X,Bilinear(conv1_1),Bilinear(conv1_2),Bilinear(conv2_1),Bilinear(conv2_2),Bilinear(conv3_1),Bilinear(conv3_2),Bilinear(conv3_3)])
    return tf.reshape(H,[input_size*input_size,feature_size])
    
    
    
def translator(H,hidden_size = 1024):
    #the 1st FCN layer
    fc1_w = weights([feature_size,hidden_size*2], name1 = 'fc1_w')
    fc1_b = bias([hidden_size*2],name2 = 'fc1_b')
    hidden1 = tf.nn.relu(tf.matmul(H,fc1_w)+fc1_b)
    
    #the 2nd FCN layer
    fc2_w = weights([hidden_size*2,hidden_size], name1 = 'fc2_w')
    fc2_b = bias([hidden_size],name2 = 'fc2_b')
    hidden2 = tf.nn.relu(tf.matmul(hidden1,fc2_w)+fc2_b)
    
    #the 3rd FCN layer 
    fc3_w1_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w1_1')
    fc3_b1_1   = bias([hidden_size/2],name2 = 'fc3_b1_1')
    fc3_w1_2   = weights([hidden_size/2,32],name1 = 'fc3_w1_2')
    fc3_b1_2   = bias([32],name2 = 'fc3_b1_2')
    hidden3_1 = tf.nn.relu(tf.matmul(hidden2,fc3_w1_1)+fc3_b1_1)
    output_X1 = tf.matmul(hidden3_1,fc3_w1_2)+fc3_b1_2
    
    fc3_w2_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w2_1')
    fc3_b2_1   = bias([hidden_size/2],name2 = 'fc3_b2_1')
    fc3_w2_2   = weights([hidden_size/2,32],name1 = 'fc3_w2_2')
    fc3_b2_2   = bias([32],name2 = 'fc3_b2_2')
    hidden3_2 = tf.nn.relu(tf.matmul(hidden2,fc3_w2_1)+fc3_b2_1)
    output_X2 = tf.matmul(hidden3_2,fc3_w2_2)+fc3_b2_2
    
    fc3_w3_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w3_1')
    fc3_b3_1   = bias([hidden_size/2],name2 = 'fc3_b3_1')
    fc3_w3_2   = weights([hidden_size/2,32],name1 = 'fc3_w3_2')
    fc3_b3_2   = bias([32],name2 = 'fc3_b3_2')
    hidden3_3 = tf.nn.relu(tf.matmul(hidden2,fc3_w3_1)+fc3_b3_1)
    output_X3 = tf.matmul(hidden3_3,fc3_w3_2)+fc3_b3_2
    
    fc3_w4_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w4_1')
    fc3_b4_1   = bias([hidden_size/2],name2 = 'fc3_b4_1')
    fc3_w4_2   = weights([hidden_size/2,32],name1 = 'fc3_w4_2')
    fc3_b4_2   = bias([32],name2 = 'fc3_b4_2')
    hidden3_4 = tf.nn.relu(tf.matmul(hidden2,fc3_w4_1)+fc3_b4_1)
    output_X4 = tf.matmul(hidden3_4,fc3_w4_2)+fc3_b4_2
    
    fc3_w5_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w5_1')
    fc3_b5_1   = bias([hidden_size/2],name2 = 'fc3_b5_1')
    fc3_w5_2   = weights([hidden_size/2,32],name1 = 'fc3_w5_2')
    fc3_b5_2   = bias([32],name2 = 'fc3_b5_2')
    hidden3_5 = tf.nn.relu(tf.matmul(hidden2,fc3_w5_1)+fc3_b5_1)
    output_X5 = tf.matmul(hidden3_5,fc3_w5_2)+fc3_b5_2
    
    fc3_w6_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w6_1')
    fc3_b6_1   = bias([hidden_size/2],name2 = 'fc3_b6_1')
    fc3_w6_2   = weights([hidden_size/2,32],name1 = 'fc3_w6_2')
    fc3_b6_2   = bias([32],name2 = 'fc3_b6_2')
    hidden3_6 = tf.nn.relu(tf.matmul(hidden2,fc3_w6_1)+fc3_b6_1)
    output_X6 = tf.matmul(hidden3_6,fc3_w6_2)+fc3_b6_2
    
    fc3_w7_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w7_1')
    fc3_b7_1   = bias([hidden_size/2],name2 = 'fc3_b7_1')
    fc3_w7_2   = weights([hidden_size/2,32],name1 = 'fc3_w7_2')
    fc3_b7_2   = bias([32],name2 = 'fc3_b7_2')
    hidden3_7 = tf.nn.relu(tf.matmul(hidden2,fc3_w7_1)+fc3_b7_1)
    output_X7 = tf.matmul(hidden3_7,fc3_w7_2)+fc3_b7_2
    
    fc3_w8_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w8_1')
    fc3_b8_1   = bias([hidden_size/2],name2 = 'fc3_b8_1')
    fc3_w8_2   = weights([hidden_size/2,32],name1 = 'fc3_w8_2')
    fc3_b8_2   = bias([32],name2 = 'fc3_b8_2')
    hidden3_8 = tf.nn.relu(tf.matmul(hidden2,fc3_w8_1)+fc3_b8_1)
    output_X8 = tf.matmul(hidden3_8,fc3_w8_2)+fc3_b8_2
    
    fc3_w9_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w9_1')
    fc3_b9_1   = bias([hidden_size/2],name2 = 'fc3_b9_1')
    fc3_w9_2   = weights([hidden_size/2,32],name1 = 'fc3_w9_2')
    fc3_b9_2   = bias([32],name2 = 'fc3_b9_2')
    hidden3_9 = tf.nn.relu(tf.matmul(hidden2,fc3_w9_1)+fc3_b9_1)
    output_X9 = tf.matmul(hidden3_9,fc3_w9_2)+fc3_b9_2
    return tf.concat_v2(axis=1,values=[output_X1,output_X2,output_X3,output_X4,output_X5,output_X6,output_X7,output_X8,output_X9]),\
                tf.concat_v2(axis=1,values=[tf.nn.softmax(output_X1),tf.nn.softmax(output_X2),tf.nn.softmax(output_X3),\
                tf.nn.softmax(output_X4),tf.nn.softmax(output_X5),tf.nn.softmax(output_X6),tf.nn.softmax(output_X7),\
                tf.nn.softmax(output_X8),tf.nn.softmax(output_X9)])
    

       
def vectorised_pol(IM):     
    temp = IM.size
    IM = np.reshape(IM,[temp/9,9])
    data_X  = np.zeros([temp/9,32*9])  
    for i in range(9):
        data_temp = np.reshape(const_array[i,:],[1,32])
        data_temp = np.tile(data_temp,[temp/9,1])
        data_temp = abs(np.transpose(np.tile(IM[:,i],[32,1]),[1,0])-data_temp)
        data_X[:,i*32:(i+1)*32] = np.equal(data_temp,np.transpose(np.tile(np.min(data_temp,axis = 1),[32,1]),[1,0]))
    return data_X


def recover_pol(data):
    data_V = np.zeros([input_size,input_size,9])
    data.shape = input_size,input_size,32*9
    for i in range(9):
        temp = np.tile(const_array[i,:],[input_size,input_size,1])
        data_V[:,:,i] = np.sum(data[:,:,i*32:(i+1)*32]*temp,2)
#        temp = const_array[i,:]
#        data_V[:,:,i] = temp[np.argmax(data[:,:,i*32:(i+1)*32],2)]                 
    return data_V
    
    
    
    