import input_data
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

class Img2Spike:
    def __init__(self, path, two_class,train_set,test_set,train_pro,test_pro):
        #导入数据
        self.mnist = self.load_data(path)
        #提取训练集，验证集、测试集
        self.train_X, self.validation_X, self.test_X, self.train_Y, self.validation_Y, self.test_Y = self.extract_data()
        #从10类中选择两类手写字符体，在初始化时传入一个list
        self.cho_train_X,  self.cho_test_X, self.cho_train_Y, self.cho_test_Y = self.choose_data(two_class,train_set,test_set)

        # 添加噪声
        for i in range(self.cho_train_X.shape[0]): # 得到训练集添加噪声图片
            self.cho_train_X[i] = self.getNoiseimg(self.cho_train_X[i],train_pro)
        for i in range(self.cho_test_X.shape[0]): # 得到测试集添加噪声图片
            self.cho_test_X[i] = self.getNoiseimg(self.cho_test_X[i], test_pro)
        pass
    
    
    def load_data(self, path):
        mnist = input_data.read_data_sets(path,one_hot=True)
        return mnist
    
    def extract_data(self):
        train_X = self.mnist.train.images                #训练集样本
        validation_X = self.mnist.validation.images      #验证集样本
        test_X = self.mnist.test.images                  #测试集样本
        #labels
        train_Y = self.mnist.train.labels                #训练集标签
        validation_Y = self.mnist.validation.labels      #验证集标签
        test_Y = self.mnist.test.labels                  #测试集标签
        '''
        train_X.shape = (55000, 784)    train_Y.shape = (55000, 10)
        test_X.shape = (10000, 784)     test_Y.shape = (10000,10)
        '''
        
        return train_X, validation_X, test_X, train_Y, validation_Y, test_Y
    
    def choose_data(self, two_class, train_set, test_set):
        #two_class 是一个长度为2的list
        cho_train_X = []
        cho_train_Y = []
        j=0
        for i in range(len(self.train_X)):
            if self.train_Y[i][two_class[0]] == 1:
                cho_train_X.append(self.train_X[i])
                cho_train_Y.append(True)
                j+=1
            elif self.train_Y[i][two_class[1]] == 1:
                cho_train_X.append(self.train_X[i])
                cho_train_Y.append(False)
                j += 1
            if j == train_set:
                break
                
        cho_test_X = []
        cho_test_Y = []
        j=0
        for i in range(len(self.test_X)):
            if self.test_Y[i][two_class[0]] == 1:
                cho_test_X.append(self.test_X[i])
                cho_test_Y.append(True)
                j+=1
            elif self.test_Y[i][two_class[1]] == 1:
                cho_test_X.append(self.test_X[i])
                cho_test_Y.append(False)
                j += 1
            if j == test_set:
                break
        '''
        cho_train_X.shape = (11623,784)
        cho_test_X.shape = (2144,784)
        '''
        return np.array(cho_train_X),  np.array(cho_test_X), cho_train_Y, cho_test_Y

    def addNoise(self,pro=0): # 0, 0.15, 0.30
        return np.random.random(size=(784,)) *pro
    def getNoiseimg(self,vec,pro): # 图片为784
        nosie = self.addNoise(pro=pro)
        for i in range(len(vec)):
            if vec[i] == 0:
               vec[i] = vec[i] +nosie[i]
        return vec

    #展示部分数据
    def show_data(self):
        fig, ax = plt.subplots(nrows=3,ncols=3,sharex='all',sharey='all')
        ax = ax.flatten()
        for i in range(9):
            img = self.cho_test_X[i].reshape(28, 28)
            ax[i].imshow(img)
            print(self.cho_test_Y[i])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()
    
    
    #第一种编码方式    
    def encode_data1(self, train_set=200,test_set=200):
        '''
        将图片向量resize为28*28的黑白图片，
        对图片做2*2的maxpooling，得到14*14的图像，
        然后补充一行一列，得到15*15的图像，
        然后按3*3划块变成一个二进制串，转换为十进制，表示为一个脉冲刺激到来时间
        一个5*5个突触
        如果一个可见域全为0， 则对应这一块不输入刺激，减少了计算量
        不全为0，则表示有信息输入，黑色块越多，表示这个可见域越明显，越提前产生刺激
        表达式为：2^9-1-value
        '''
        X = self.generate_batch(train_set, self.cho_train_X)
        Y = self.cho_train_Y[0:train_set]
        train_spike = []
        for i in range(train_set):
            spike_time = self.generate_spike(X[i], Y[i])
            train_spike.append(spike_time)

        X = self.generate_batch(test_set, self.cho_test_X)
        Y = self.cho_test_Y[0:test_set]
        test_spike = []
        for i in range(test_set):
            spike_time = self.generate_spike(X[i], Y[i])
            test_spike.append(spike_time)

        return train_spike, test_spike

    
    #生成一张图片的脉冲序到来时间，并将其和标签（bool值）组成一个元组放到总脉冲列表中
    def generate_spike(self, X, Y):
        spike_time = []
        for j in range(5):
            for k in range(5):
                bin_str = X[j, :, k, :].flatten()
                value = self.bin2dec(bin_str)
                if value == 0:
                   spike_time.append([])
                else:
                    spike_time.append([2**9 - value])
        spike_time = (np.array(spike_time), Y)  # 把脉冲信号和标签组合成一个元组

        return spike_time
    
    #二进制序列转变为十进制
    def bin2dec(self, bin_str):
        flag = 0
        for i in range(len(bin_str)):
            flag= bin_str[i]* 2**i + flag
            
        return flag
    
    #第二种编码方式
    def encode_data2(self, train_set=200,test_set=200):
        '''
        将图片向量resize为28*28的黑白图片，
        对图片做2*2的maxpooling，得到14*14的图像，
        然后补充一行一列，得到15*15的图像，
        然后按3*3划块变成一个二进制串，转换为十进制，表示为一个脉冲刺激到来时间
        一个5*5个突触
        '''
        #返回为一个张量X.shape=[train_set,5,3,5,3]
        X = self.generate_batch(train_set,self.cho_train_X)
        Y = self.cho_train_Y[0:train_set]
        train_spike = []
        for i in range(train_set):
            spike_time = self.generate_spike2(X[i], Y[i])
            train_spike.append(spike_time)
            
        X = self.generate_batch(test_set,self.cho_test_X)
        Y = self.cho_test_Y[0:test_set]
        test_spike = []
        for i in range(test_set):
            
            spike_time = self.generate_spike2(X[i], Y[i])
            test_spike.append(spike_time)
        
        return train_spike, test_spike        
    
    def generate_batch(self, data_set, X):
        #池化
        X = np.round(X[0:data_set])
        X = torch.from_numpy(X)
        img = torch.reshape(X,(data_set,28,28))
        maxpooling = torch.nn.MaxPool2d((2, 2), stride=(2, 2))
        img = maxpooling(img)
        #按3*3划块
        new_img = torch.zeros(data_set,15,15)
        new_img[:,0:14,0:14] = img
        new_img = new_img.reshape(data_set,5,3,5,3)
        
        return new_img 
    
    
    def generate_spike2(self, X, Y):
        spike_time = []

        for j in range(5):
            for k in range(5):  
                bin_str = X[j,:,k,:].flatten()
                value = self.bin2dec(bin_str)
                spike_time.append([value])
        spike_time = (np.array(spike_time), Y) #把脉冲信号和标签组合成一个元组
        
        return spike_time
    
    
    def encode_data3(self, train_set=200,test_set=200):
        '''
        将图片向量resize为28*28的黑白图片,
        然后按可见域为3*3划块变成一个二进制串，转换为十进制，表示为一个脉冲刺激到来时间，
        逐一向左、向下平移，28*28变为26*26
        如果一个可见域全为0， 则对应这一块不输入刺激，减少了计算量
        不全为0，则表示有信息输入，黑色块越多，表示这个可见域越明显，越提前产生刺激
        表达式为：2^9-1-value
        '''
        #返回为一个张量X.shape=[train_set,14,14]
        X = self.generate_batch2(train_set,self.cho_train_X)
        Y = self.cho_train_Y[0:train_set]
        train_spike = []
        for i in range(train_set):
            spike_time = self.generate_spike3(X[i], Y[i])
            train_spike.append(spike_time)
            
        X = self.generate_batch2(test_set,self.cho_test_X)
        Y = self.cho_test_Y[0:test_set]
        test_spike = []
        for i in range(test_set):
            spike_time = self.generate_spike3(X[i], Y[i])
            test_spike.append(spike_time)
        
        return train_spike, test_spike    
        
        
    def generate_batch2(self, data_set,X):
        X = np.round(X[0:data_set]+0.4)
        X = torch.from_numpy(X)
        img = torch.reshape(X,(data_set,28,28))
        maxpooling = torch.nn.MaxPool2d((2, 2), stride=(2, 2))
        img = maxpooling(img)#按14*14的图片
        
        return img
    
    def generate_spike3(self, X, Y):
        spike_time = []

        for j in range(2,10):
            for k in range(2,10):  
                bin_str = X[j:j+3,k:k+3].flatten()
                value = self.bin2dec(bin_str)
                if value == 0:
                    spike_time.append([])
                else:
                    spike_time.append([2**9 -1 - value])
        spike_time = (np.array(spike_time), Y) #把脉冲信号和标签组合成一个元组
        
        return spike_time


if __name__ == "__main__":
    img2spike = Img2Spike("fashion/",[0,1])
    A = img2spike.Noise()
    print(A)

