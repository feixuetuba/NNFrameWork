import math

class Sigmoid:
    def __init__(self, alpha=0):
        pass
    def forward(self, x):
        return 1.0 / (1.0 + pow(math.e, -x))
    def backward(self, x):
        return x * (1 - x)

class PRelu:
    def __init__(self, alpha=0):
        self.alpha = alpha
        
    def forward(self, x):
        if x >= 0:
            return x
        else:
            return self.alpha * x
    
    def backward(self, x):
        if x > 0:
            return 1
        else:
            return -self.alpha