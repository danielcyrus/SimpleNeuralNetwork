import numpy as np

class Network(object):
    def __init__(self, layers, learningRate=0.001):
        self.layers = layers
        self.learningRate = learningRate
        self.weights = [np.random.normal(0.0, pow(2, -0.5),(node , synapse)) for (node , synapse) in zip(layers[:-1],layers[1:])]
        self.s_outputs = [np.zeros(z) for z in layers[1:]]#include hidden layers outputs # this output store sigmoid output
       
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x)) 
    
    def softmax_grad(self,x):
        s = x.reshape(-1,1)
        return np.diagflat(x) - np.dot(x, x.T)

    def sigmoid(self,x):
     
        return (1.0 / ( 1.0 + np.exp(-x)))
    
    def derevative_sigmoid(self,x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def feedForward(self, inputs):
        # output = sigmoid(wieght * inputs)
        _inputs = inputs.copy()
        for i in range(len(self.layers)-1):
            temp = np.dot(_inputs,self.weights[i])
            self.s_outputs[i]=self.sigmoid(temp)
            _inputs = self.s_outputs[i]

    def backPropagation(self, inputs, expected_output):
        #1-last error = expected - current
        #2-hidden_error = [weights].T*[error]
        #3-back propagation equation:
        #   newWeight = oldWeight + learningRate * (error * derivative_sigmoid( dot(oldWeight * prev_outputs) ) * prev_outputs.T)
        error = expected_output - self.s_outputs[len(self.s_outputs)-1]

        for i in range(len(self.s_outputs)-1, 0 , -1):
            if(i-1>=0):
                sigO = np.dot(self.weights[i].T,self.s_outputs[i-1])
                prevOut = self.s_outputs[i-1]
            else:
                sigO = np.dot(self.weights[i].T,inputs)
                prevOut = inputs
            derivative = self.derevative_sigmoid(sigO)
            hidden_error = np.dot(self.weights[i] , error)
            self.weights[i] += self.learningRate * np.dot(np.matrix(prevOut).T,np.matrix(error * derivative))
            error = hidden_error
            
    def train(self, inputs, outputs, epoch):
        for i in range(epoch):
            self.feedForward(inputs)
            self.backPropagation(input,outputs)

    
    def predict(self, input):
        self.feedForward(input)
        return np.argmax(self.s_outputs[len(self.s_outputs)-1])





        





