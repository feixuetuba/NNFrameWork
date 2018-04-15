#-*- coding:utf-8 -*-
import pickle

class NeuralNetwork:
    def __init__(self, input_num, lr, activations, loss_obj, initor_obj):
        #layers:[{'values':[], 'weights':[[W11, W21, W31..],[W12, W22, W32]], 'bias':[[b11, b21..].[]]}]
        self.layers = [{'name':'input', 'num': input_num}]   
        self.inputs = None
        self.outputs = None
        self.labels = None
        self.lr = lr
        self.total_loss = 0
        self.init_func = initor_obj.genRandom
        if type(activations) != dict:
            raise Exception('activation should be a dict like {"prelu":prelu_obj}')
        self.activations = activations

        self.loss_func = loss_obj
        
    def add_layer(self, name, neural_count, activation='sigmoid', weights=None, bias=None):
        num = self.layers[-1]['num']
        layer = {}
        layer['weights'] = weights if weights is not None else self.init_func(neural_count, num)
        layer['bias'] = bias if bias is not None else [0 for i in range(neural_count)]
        layer['name'] = name
        layer['num'] = neural_count
        layer['activation'] = activation
        if activation not in self.activations:
            print("no such activation function")
            exit()
        self.layers.append(layer)
    
    def train(self, feed_dict):
        self.labels = feed_dict['labels']
        self.inputs = feed_dict['inputs']
        if len(self.inputs) != self.layers[0]['num']:
            raise Error('invalid input')
        self.__forward()
        self.__cal_loss()
        self.__backward()
    
    def estimate(self, feed_dict):
        self.inputs = feed_dict['inputs']
        if len(self.inputs) != self.layers[0]['num']:
            raise Error('invalid input')
        self.__forward()
    
    def dump_model(self, file_path):
        with open(file_path, 'wb') as fd:
            pickle.dump(self.layers, fd, protocol=2)
    
    def load_model(self, file_path):
        with open(file_path, 'rb') as fd:
            self.layers = pickle.load(fd)
        for layer in self.layers[1:]:
            if 'act' not in layer:
                layer['activation'] = "sigmoid"

    def __forward(self):
        values = self.inputs
        self.layers[0]['in'] = values
        self.layers[0]['out'] = values
        #print("******",self.total_loss, '******')
        for layer in self.layers[1:]:
            neural_count = layer['num']
            activation = self.activations[layer['activation']]
            layer['in'] = values     #记录上一层的输出
            values = []
            for i in range(neural_count):
                value = 0
                 #依次取出上层第n个节点的输出，与n到本层第i个节点的权值
                for v,w in zip(layer['in'],layer['weights'][i]):
                    value += w * v
                value += layer['bias'][i]
                values.append(activation.forward(value))
                
            #当前层的输出即下一层的输入
            layer['out'] = values
        '''    
            print (layer['name'])
            print ('\tinput', layer['in'])
            print ('\tweights', layer['weights'])
            print ('\tbias', layer['bias'])
            print ('\toutput', layer['out'] )
        print('\tlabeel',self.labels)    
        '''
        self.outputs = self.layers[-1]['out']
    def __backward(self):
        layer_losses = self.layers[-1]['out_losses']
        for layer in self.layers[-1:0:-1]:
            activation = self.activations[layer['activation']]
            loss_sum = 0
            out_loss = layer_losses
            losses = []
            _weights = []
            #更新当前层的权值
            neural_losses = []      #记录隐层的loss分量
            bias = []
            layer_losses = []
            #weights=[[W11, W21, W31],[W12, W22, W32]...]
            for i in range(layer['num']):
                neural_loss = out_loss[i]
                weights = layer['weights'][i]
                output = layer['out'][i]
                
                layer_loss_vector = []
                weight_update_vector = []
                bias_update_vector = []
                
                diff_output = activation.backward(output)
                
                #weights
                for v,w in zip(layer['in'], weights):
                    weight_update_vector.append(
                        self.lr * neural_loss * diff_output * v)  #slop of w = E(n+1) * diff_o(n) * In(n-1
                
                #bias
                bias_update_vector.append(              #slop of b = E(n+1) * diff_o(n) 
                    neural_loss * diff_output
                )
                
                #layer losses
                losses_vector = []
                for w in weights:
                    losses_vector.append(
                        neural_loss * diff_output * w
                    )
                losses.append(losses_vector)
                #更新
                #weights
                #print("slops:",weight_update_vector)
                for i, w_slop in enumerate(weight_update_vector):
                    weights[i] -= w_slop
                
                #bias
                for i, b_slop in enumerate(bias_update_vector):
                    layer['bias'][i] -= self.lr * b_slop
                
            #layer loss
                w_count = len(losses_vector)
                for i in range(w_count):
                    neural_loss = 0
                    for losses_vector in losses:
                        neural_loss += losses_vector[i]
                    layer_losses.append(neural_loss)
            
        #for layer in self.layers[1:]:
        #    print(layer['name'], layer['weights'], layer['bias'],)
    
    def __cal_loss(self):
        layer = self.layers[-1]
        losses = []
        self.total_loss = 0
        if len(self.outputs) != len(self.labels):
            raise Exception("length of outputs unequal to  length of labels")
        for o, l in zip(self.outputs, self.labels):
            loss = self.loss_func.forward(l, o)
            losses.append(self.loss_func.backward(l, o))
            self.total_loss += loss
        layer['out_losses'] = losses    

if __name__ == '__main__':
    from estimators import MSELoss as MSE
    from activations import PRelu, Sigmoid
    from initors import GaussInitor

    import os
    TRAIN = True
    model_file = './nnmode.ckpt'
    relu = PRelu(0)
    sigmoid = Sigmoid(0)
    mse = MSE()
    initor = GaussInitor()
    activations = {'prelu': relu}
    NET = NeuralNetwork(
        input_num=2, lr=0.01, 
        activations=activations, loss_obj=mse, 
        initor_obj = initor)
    if os.path.isfile(model_file) and not TRAIN:
        NET.load_model(model_file)
    else:
        NET.add_layer('hidden', 2, activation='prelu',
                #,weights=[[0.15, 0.2],[0.25, 0.3]],
                #bias=[0.35, 0.35]
                )
        NET.add_layer('output', 2, activation='prelu'
                #,weights=[[0.4, 0.45],[0.5, 0.55]],
                #bias=[0.6, 0.6]
                )
        #NET.dump_model(model_file)
    #NET.train(feed_dict={'inputs':[0.05, 0.1], 'labels':[0.01, 0.99]})
    print (NET.layers)
    for x in range(100):
        NET.train(feed_dict={'inputs':[0.05, 0.1], 'labels':[0.01, 0.99]})
        print (NET.outputs, NET.total_loss)
    