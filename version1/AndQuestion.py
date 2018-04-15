#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from myframework.nncraft import NeuralNetwork as NNetWork
from myframework.estimators import MSELoss as MSE
from myframework.activations import PRelu
from myframework.initors import GaussInitor

if __name__ == '__main__':
    relu = PRelu(0)                 #激活函数， 除了这个还可以选择sigmoid，目前只实现了这两个
    mse = MSE()                     #损失估计函数，目前只实现了这个
    initor = GaussInitor(u=0.2, v=0.1)  #神经元参数初始化工具，除了高斯之外，还实现了XavierInitor
    activations = {'prelu': relu}       #激活函数列表，网络的每一层允许使用不同的激活函数，所以一开始必须向框架注册这些..
    nn = NNetWork(                  #创建框架对象，这个没啥说的
        input_num=2, lr=0.01, 
        activations=activations, loss_obj=mse, 
        initor_obj = initor)
    
    #w1 = 0.0001
    #w2 = -0.0001
    #b = 0.0001
    nn.add_layer(                       #为模型添加一个网络层
        name='neural',
        neural_count = 1,
        activation = 'prelu'
        #,weights = [[w1, w2]],         #这里可以手动的指定模型的初始化参数
        #bias = [b]
    )
    
    import datetime, os
    #ckpt_file = "%s.ckpt"%datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    #ckpt_file = os.path.join(r'D:\ZM\playground\3\init_ana', ckpt_file)
    #nn.dump_model(ckpt_file)
    data = [[0, 0, 0], [0, 1, 0],
        [1, 0, 0], [1, 1, 1]]
    
    #layer = nn.layers[1]
    #ws = layer['weights'][0]
    for epoch in range(50):
        for d in data:
            feed_dict = {
                'inputs': [d[0], d[1]],
                'labels': [d[2]]
            }
            nn.train(feed_dict=feed_dict)
            layer = nn.layers[1]
            #ws = layer['weights'][0]
            print("total_loss:%.3f"%nn.total_loss)
    #estimate
    threshold = 0.5                 #模型输出是[0-1],直观的感觉采用0.5为分界线，你可根据实际需要去改
    for d in data:
        feed_dict = {
                'inputs': [d[0], d[1]],
                'labels': [d[2]]
            }
        nn.estimate(feed_dict=feed_dict)
        Y = nn.outputs[0]
        if Y < threshold:
            plt.plot(d[0], d[1], 'ro')
        else:
            plt.plot(d[0], d[1], 'bo')
    print("end")
    plt.show()