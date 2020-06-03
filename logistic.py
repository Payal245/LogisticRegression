import numpy as np
class logistic:
    def __init__(self,lr=0.0001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.base=None
        self.weight=None

    def fit(self,x,y):
        n_samples=x.size
        self.weight=np.zeros(n_samples)
        self.bias=0

        for _ in range(self.n_iters):
            linear_model=np.dot(x,self.weight)+self.bias
            y_predict=self._sigmoid(linear_model)
            dw=(1/n_samples) * np.dot(x.T, (y_predict-y))
            db=(1/n_samples) * np.sum(y_predict - y)

            self.weight =self.weight - self.lr * dw
            self.bias =self.bias - self.lr * db

    def predict(self,x):
        linear_model=np.dot(x,self.weight)+self.bias
        y_predict=self._sigmoid(linear_model)
        y_predict_cls=[1 if i >0.51 else 0 for i in y_predict]
        return y_predict_cls[0]

    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))