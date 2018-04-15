import matplotlib.pyplot as plt
from myframework.nncraft import NeuralNetwork as NNetWork
from myframework.estimators import MSELoss as MSE
from myframework.activations import PRelu, Sigmoid
from myframework.initors import GaussInitor, XavierInitor

if __name__ == '__main__':
    relu = PRelu(0)
    sigmoid = Sigmoid(0)
    mse = MSE()
    initor = GaussInitor(u=0.2, v=0.1)
    #initor = XavierInitor()
    activations = {'relu': relu}
    nn = NNetWork(
        input_num=2, lr=0.1, 
        activations=activations, loss_obj=mse, 
        initor_obj = initor)
    
    nn.add_layer(
        name='hidden1',
        neural_count = 2,
        activation = 'relu'
    )
    
    nn.add_layer(
        name='output',
        neural_count = 1,
        activation = 'relu',
    )
    
    for layer in nn.layers:
        print(layer)
    
    data = [[0, 0, 0], [0, 1, 1],
        [1, 0, 1], [1, 1, 0]]
    
    layer = nn.layers[1]
    ws = layer['weights'][0]
    for epoch in range(2000):
        for d in data:
            feed_dict = {
                'inputs': [d[0], d[1]],
                'labels': [d[2]]
            }
            nn.train(feed_dict=feed_dict)
            #print(nn.total_loss)
    
    #estimate
    threshold = 0.5
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