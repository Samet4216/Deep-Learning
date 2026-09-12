"""
def neuron(x1,x2,x3,w1,w2,w3,b):
    z=(x1*w1)+(x2*w2)+(x3*w3)+b
    return z

input=neuron(
    2,#x1
    3,#x2
    4,#x3
    5,#w1
    6,#w2
    0,#w3
    20,#b
)
print(input)
"""

import numpy as np
import random 
import activation_functions as af

class neuron():

    def __init__(self, input_size, activation, derivate):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.1 #bias başlangıçta negatif olursa relu ölür
        self.activation = getattr(af, activation) #getattr iki parametre alır, birinci parametre modül ismi, ikinci parametre fonksiyon ismi string olarak verilir.
        self.derivate = getattr(af, derivate) #activation ve derivate isimleri string olarak verilir.


    def forward(self,input):
        self.z = np.dot(input,self.weights)+self.bias #z = (x*w)+b
        self.output = self.activation(self.z)
        return self.output

    #def mse_loss(y_true, y_pred):

    def gradient(self, x, y_true):

        y_pred = self.output #aktivasyon fonksiyonundan gelen tahmin

        dL_dy_pred = -2 * (y_true - y_pred) #loss fonkun sigmoid çıktısına göre türevi
        activation_func_der = self.derivate(self.z)

        # Chain rule
        ortak_turevler = dL_dy_pred * activation_func_der
        gradient = ortak_turevler * x #(dz_dw)

        bias_gradient = ortak_turevler * 1 #dz_db türevi b katsayısı 1 

        return gradient, bias_gradient

    def update(self, gradient, bias_gradient, learning_rate):
        self.weights -= learning_rate * gradient
        self.bias -= learning_rate * bias_gradient


def mse_loss(y_true, y_pred):
    loss=(y_true-y_pred)**2
    return loss