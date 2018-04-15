#-*- coding:utf-8 -*-
import random
import math

class GaussInitor:
    '''
    vtype = 'std', use u as mean value, v as stddev
    vtype = 'Xavier1', stddev = sqrt(1/input) 
    vtype = 'Xavier2', stddev = sqrt(2/(input+output)) 
    vtype = 'MSRA', stddev = sqrt(2/input) 
    '''
    def __init__(self, u=1, v=0.1, vtype="std"):
        #if theata <= 0:
        #    raise Exception("theata must bigger then 0")
        #self.theata = math.sqrt(theata)
        self.u = u
        self.v = v
        self.vtype = type
        
    def genRandom(self, rol , col):
        if col <= 0 :
            raise Exception('col could not be 0')
        
        v = self.v
        if self.vtype == 'Xavier1':
            v = math.sqrt(1./col)
        elif self.vtype == 'Xavier2':
            v = math.sqrt(2./(rol + col))
        elif self.vtype == 'MSRA':
            v = math.sqrt(2./col)
            
        if col == 1:
            if rol == 0:
                return [random.gauss(self.u, v)]
            elif rol == 1:
                return [[random.gauss(self.u, v)]]
        return [ [random.gauss(self.u, v) for c in range(col)] for r in range(rol)]

#cite: 《Understanding the difficulty of training deep feedforward neural networks》
class XavierInitor:
    def genRandom(self, rol, col):
        v = math.sqrt(6)/math.sqrt(rol + col)
        if col <= 0 :
            raise Exception('col could not be 0')
        if col == 1:
            if rol == 0:
                return [random.uniform(-v, v)]
            elif rol == 1:
                return [[random.uniform(-v, v)]]
        return [ [random.uniform(-v, v) for c in range(col)] for r in range(rol)]
 
        
if __name__ == '__main__':
    g =GaussInitor(1, 0.001)
    num_count = 3
    rol = 0
    col = 10
    values = g.genRandom(rol, col)
    v = []
    for lines in values:
        if rol > 0:
            for col in lines:
                v.append(col)
        else:
            v.append(lines)
    count = sum(v)
    avg = count/num_count
    var = sum([pow(x-avg, 2) for x in v]) / num_count
    print("avg:%f, var:%f"%(avg, var))
    print(values)
    
    xav =  XavierInitor()
    print(xav.genRandom(10, 10))
