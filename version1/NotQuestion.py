#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from myframework.nncraft import NeuralNetwork as NNetWork
from myframework.estimators import MSELoss as MSE
from myframework.activations import PRelu, Sigmoid
from myframework.initors import GaussInitor

if __name__ == '__main__':
    prelu = PRelu(0.00)                 #激活函数， 除了这个还可以选择relu，目前只实现了这两个
    mse = MSE()                     #损失估计函数，目前只实现了这个
    initor = GaussInitor(u=1, v=0.0001)  #神经元参数初始化工具，除了高斯之外，还实现了XavierInitor
    activations = {'prelu': prelu}       #激活函数列表，网络的每一层允许使用不同的激活函数，所以一开始必须向框架注册这些..
    nn = NNetWork(                  #创建框架对象，这个没啥说的
        input_num=1, lr=0.1, 
        activations=activations, loss_obj=mse, 
        initor_obj = initor)
    
    #w1 = 0.0001
    #w2 = -0.0001
    #b = 0.0001
    nn.add_layer(                       #为模型添加一个网络层
        name='neural',
        neural_count = 1,
        activation = 'prelu',
        weights = [[0.5]],
        bias = [0.1]
    )

    data = [[0, 0.99], [1, 0.001]]      #想想看，为什么不是：data = [[1, 0], [0, 1]]
    

    for epoch in range(25):
        for d in data:
            feed_dict = {
                'inputs': [d[0]],
                'labels': [d[1]]
            }
            nn.train(feed_dict=feed_dict)
            print("total_loss:%.3f"%nn.total_loss)
    #estimate
    threshold = 0.5                 #模型输出是[0-1],直观的感觉采用0.5为分界线，你可根据实际需要去改

    for d in data:
        feed_dict = {
                'inputs': [d[0]],
                'labels': [d[1]]
            }
        nn.estimate(feed_dict=feed_dict)
        Y = nn.outputs[0]
        if Y < threshold:
            plt.plot(d[0], 0, 'ro')
        else:
            plt.plot(d[0], 1, 'bo')
    print("end")
    plt.show()