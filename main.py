from Tempotron import Tempotron
from QTempotron import QTempotron
import numpy as np
from Img2Spike import Img2Spike
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


'''图片所需的参数定义'''
train_set = 200 #训练集数量
test_set = 500#测试集数量
str_len = 9 # 二进制串长度
train_pro = 0.5
test_pro = 0.5
# img2spike = Img2Spike("MNIST_data/",[1,0],train_set,test_set,train_pro,test_pro)
img2spike = Img2Spike("fashion/",[1,0],train_set,test_set,train_pro,test_pro)
img2spike.show_data()
"""
#train_spike,test_spike = img2spike.encode_data3(train_set, test_set)
train_spike,test_spike = img2spike.encode_data2(train_set, test_set)
# train_spike,test_spike = img2spike.encode_data1(train_set, test_set)
synapse_num = train_spike[0][0].shape[0] #突触数量
print("编码完成")

'''执行Tempotron所需的参数定义'''
steps = 50 #迭代次数
efficacies = np.random.random(synapse_num)#产生的是一个伪随机数，每次运行的结果都不会变
#print('synaptic efficacies:', efficacies, '\n')
learning_rate = 0.003

'''执行Temportorn'''
#__init__(self, V_rest, tau, tau_s, train_set,test_set. steps, synaptic_efficacies):
# tempotron = QTempotron(0, 20, 5, train_set, test_set, steps, efficacies,str_len)
tempotron = Tempotron(0, 20, 5, train_set, test_set, steps, efficacies,str_len)

for step in range(steps):
    train_percentage = tempotron.train(train_spike, learning_rate, step)
    test_percentage = tempotron.test(test_spike, step)
    
    print("第",step,"步的","训练集数据准确率：",np.round(train_percentage,3),"测试集数据准确率：", np.round(test_percentage,3))
    # print('synaptic efficacies:', efficacies, '\n')


s = np.linspace(0,steps-1,steps)
plt.plot(s,tempotron.percentage,s,tempotron.test_percentage)
plt.legend(labels=['training set', 'test set'],loc='best')
plt.savefig('classification_accuracy.jpg',dpi=720)
"""