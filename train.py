# -*- coding: utf-8 -*-
"""
main.py
Code for paper: Code for Paper Q. Song, F. Xu, and Y.Q. Jin, 
"Radar Image Colorization: Converting Single-Polarization to 
Fully Polarimetric Using Deep Neural Networks," IEEE Access.

SAR Image Colorization: Reconstruction full-pol data from single-pol data.
by Qian Song on Feb. 26th 2017

"""

import utils
import numpy as np
import time
import scipy.io as sio
import h5py
import tensorflow as tf
import random

class SARIC(object):
    def __init__(self, sess):
        self.sess = sess
        
        self.learning_rate = 0.0001
        self.output_size = 400
        self.training_size = 19
        self.total_size = 189
        self.feature_size = 1153
        self.test_size = 4
        self.re_total_size = 135

        self.model_build()
        
    def model_build(self):
        
        ##Build Model:=====================================================================
        self.VVVV = tf.placeholder(tf.float32,[None,self.output_size,self.output_size,1])  #Single-pol input data
        self.H = tf.placeholder(tf.float32,[None,self.feature_size])                       #Spatial features
        self.X_true = tf.placeholder(tf.float32,[None,32*9])
        
        self.hypercolumn = utils.VGG16(self.VVVV)      #Extracted spatical features from VGG16
        self.X,self.X_ = utils.translator(self.H)
        
        self.d_loss = - tf.reduce_mean(self.X_true*tf.log(self.X_+1e-7))
        self.optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate, epsilon = 1e-6) \
                                  .minimize(self.d_loss)
        self.saver = tf.train.Saver()

    def train(self):
        
        init = tf.initialize_all_variables()
        self.sess.run(init)
        start_time = time.time()
        
        #Load the Normalization Parameters:=======================
        matfn = './samples/mean_and_var.mat'
        data1 = sio.loadmat(matfn)
        temp_mean = data1['mean']
        temp_mean = np.mean(temp_mean,axis=0)
        temp_var  = data1['var']
        temp_var[temp_var<1.0] = 1.0
        temp_var = np.mean(temp_var,axis=0)
        #
        
#        #Load Checkpoint
#        print("[*]Loading Model...")
#        self.saver.restore(self.sess, "./checkpoint/SARColorization1")
#        print("[*]Load successfully!")
        

        self.is_train = True
        data_X,data_V = self.load_data()

        #Training steps:=====================================
        counter = 0
        temp_list1 = np.linspace(0,self.output_size*self.output_size-1,self.output_size*self.output_size,dtype = 'int')
        temp_list2 = np.linspace(0,self.training_size-1,self.training_size,dtype = 'int')
        
        for epoch in range(100):
            batch_idxs = len(data_X)
            random.shuffle(temp_list2)
            random.shuffle(temp_list1)
            for idx in temp_list2:                
                batch_V = np.reshape(data_V[idx,:,:],[1,self.output_size,self.output_size,1])
                temp_H  = self.sess.run([self.hypercolumn],feed_dict = {self.VVVV:batch_V})
                temp_H = temp_H[0]
                temp_H = (temp_H-temp_mean)/temp_var                
                temp_X = utils.vectorised_pol(data_X[idx,:,:,:])
                
                for index in range(80):                                       
                    batch_H = temp_H[temp_list1[index*2000:(index+1)*2000],:]
                    batch_X = temp_X[temp_list1[index*2000:(index+1)*2000],:]                
                    loss_,train_step,X = self.sess.run([self.d_loss, self.optim,self.X_], feed_dict={self.H: batch_H, self.X_true:batch_X})
                
                    counter += 1
                    if np.mod(counter,10)==9:
                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f" \
                                    % (epoch, idx+1, batch_idxs,
                                        time.time() - start_time,loss_))                                          
                     
            self.saver.save(self.sess,"./checkpoint/SARColorization1")
            print("[*]Save Model...")
            
        #Prediction of Test1:=========================================================
        Re_data = np.zeros([self.total_size,self.output_size,self.output_size,9])
        for i in range(self.total_size):
            batch_V = np.reshape(data_V[i,:,:],[1,self.output_size,self.output_size,1])
            batch_H = self.sess.run([self.hypercolumn],feed_dict={self.VVVV:batch_V})
            batch_H = batch_H[0]
            batch_H = (batch_H-temp_mean)/temp_var
            val_X   = self.sess.run([self.X_],feed_dict={self.H: batch_H})
            Re_data[i,:,:,:] = utils.recover_pol(val_X[0])
        sio.savemat('./samples/test_SD1_re0526.mat',{'Re_data':Re_data})
        
        
        self.is_train = False
        data_X,data_V,test_V = self.load_data()
                
        #Prediction of Test2:=========================================================
        Re_data = np.zeros([self.re_total_size,self.output_size,self.output_size,9])
        for i in range(self.re_total_size):
            batch_V = np.reshape(data_V[i,:,:],[1,self.output_size,self.output_size,1])
            batch_H = self.sess.run([self.hypercolumn],feed_dict={self.VVVV:batch_V})
            batch_H = batch_H[0]
            batch_H = (batch_H-temp_mean)/temp_var
            val_X   = self.sess.run([self.X_],feed_dict={self.H: batch_H})
            Re_data[i,:,:,:] = utils.recover_pol(val_X[0])
        sio.savemat('./samples/test_SD2_re0526.mat',{'Re_data':Re_data})
        
        #Truth of Test1:=========================================================
        temp_X = utils.vectorised_pol(data_X)
        data_q = np.zeros([self.total_size,self.output_size,self.output_size,9])
        for i in range(self.total_size):
            data_q[i,:,:,:] = utils.recover_pol(temp_X[i*self.output_size*self.output_size:(i+1)*self.output_size*self.output_size,:])
        sio.savemat('./samples/SD1_quan.mat',{'data_q':data_q})
        


#        #Test data:========================================
        data_test = np.zeros([self.test_size,self.output_size,self.output_size,9])
        for i in range(self.test_size):
            test_batch_V = np.reshape(test_V[i,:,:],[1,self.output_size,self.output_size,1])
            batch_H = self.sess.run([self.hypercolumn],feed_dict={self.VVVV:test_batch_V})
            batch_H = batch_H[0]
            batch_H = (batch_H-temp_mean)/temp_var
            test_X   = self.sess.run([self.X_],feed_dict={self.H: batch_H})
            data_test[i,:,:,:] = utils.recover_pol(test_X[0])
        sio.savemat('./samples/test_MN150605_VVVV_re.mat',{'Re_data':data_test})
        

    def load_data(self):
        matfn = './data/train_SD1109.mat'
        data1 = h5py.File(matfn,'r')
        data = data1['data']           
        data = np.transpose(data,axes = [3,2,1,0])
        data.shape = self.total_size,self.output_size,self.output_size,-1
        
        data_V = data1['VVVV']
        data_V = np.transpose(data_V,axes = [2,1,0])
        data_V = np.log10(data_V)*10
        data_V[data_V>0] = 0
        data_V[data_V<-25] = -25
        data_V = (data_V+25)/25
        data_V.shape = self.total_size,self.output_size,self.output_size
        
        if self.is_train == True:
            data_selected = np.zeros([self.training_size,self.output_size,self.output_size,9])
            data_selected[0:14,:,:,:]  = data[1:15,:,:,:]
            data_selected[14:18,:,:,:] = data[90:94,:,:,:]
            data_selected[18,:,:,:]    = data[78,:,:,:]
            
            dataV_selected = np.zeros([self.training_size,self.output_size,self.output_size])
            dataV_selected[0:14,:,:]  = data_V[1:15,:,:]
            dataV_selected[14:18,:,:] = data_V[90:94,:,:]
            dataV_selected[18,:,:]    = data_V[78,:,:]

            return data_selected,dataV_selected
            
        else:            
            matfn = './data/test_NO120702_VVVV.mat'
            data1 = sio.loadmat(matfn)
            test_V = data1['VVVV']
            test_V = np.log10(test_V)*10
            test_V[test_V>0] = 0
            test_V[test_V<-25]  = -25
            test_V = (test_V + 25)/25
            
            
            matfn = './data/test_SD2_VVVV.mat'
            data1 = sio.loadmat(matfn)
            data_V = data1['VVVV']
            data_V[data_V==0] = 1
            data_V = np.log10(data_V)*10
            data_V[data_V>0] = 0
            data_V[data_V<-25] = -25
            data_V = (data_V+25)/25
            
            return data,data_V,test_V       
        
def main(_):
    with tf.Session() as sess:
        SAR_image_color= SARIC(sess)
    SAR_image_color.train()
        
if __name__ == '__main__':
    tf.app.run() 
