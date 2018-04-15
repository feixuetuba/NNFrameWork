class MSELoss:
    def forward(self, y, _y):
        return 1 / 2.0 *  pow(y - _y, 2)
        
    def backward(self, y, _y):
        return _y - y
