import numpy as np
from numpy.random import randn

#create a standard RNN
class RNN:
#initialize randomized weight matrices for input, internal, and output layers, as well as input and output biases
#initial variance of weights can be accounted for in a few ways. In a limited RNN such as this one, simply dividing by 1000 to reduce initial variance works fine. However offcial methods like Xavier Initialization (detailed in the commented out code) is preferred
    def __init__(self, input_size, output_size, hidden_size = 64):
        self.input_weight = randn(hidden_size, input_size) /1000  #* np.sqrt(1 / input_size)
        self.internal_weight = randn(hidden_size, hidden_size) /1000 #* np.sqrt(1 / hidden_size)
        self.output_weight = randn(output_size, hidden_size) /1000 #* np.sqrt(1 / hidden_size)

        self.input_bias = np.zeros((hidden_size, 1))
        self.output_bias  = np.zeros((output_size, 1))
#define feedforward function
#Hidden_Feed is initialized for use as an initial internal state 
#inputs_log and hidden_log are created for cacheing input and internal values for later backpropogation 
    def feedforward(self, inputs):

        Hidden_Feed = np.zeros((self.internal_weight.shape[0], 1))

        self.inputs_log = inputs
        self.hidden_log = {0: Hidden_Feed}

        for index, vector in enumerate(inputs):
            Hidden_Feed = np.tanh(self.input_weight @ vector + self.internal_weight @ Hidden_Feed + self.input_bias)
            self.hidden_log[index + 1] = Hidden_Feed
            

        output  = self.output_weight @ Hidden_Feed  + self.output_bias

        return output, Hidden_Feed

#define our backpropogation function using cross entropy loss
#a basic knowledge of derivatives and chain rule are expected to understand this code 
    def backpropogation(self, D_Y, learning_rate = 2e-2):
        n = len(self.inputs_log)

        d_output_weight = D_Y @ self.hidden_log[n].T
        d_output_bias = D_Y

        d_input_weight = np.zeros(self.input_weight.shape)
        d_internal_weight = np.zeros(self.internal_weight.shape)
        d_input_bias = np.zeros(self.input_bias.shape)

        d_Hidden_Feed = self.output_weight.T @ D_Y

        for t in reversed(range(n)):
            temp = ((1 - self.hidden_log[t + 1] ** 2) * d_Hidden_Feed)
            d_input_bias += temp
            d_internal_weight += temp @ self.hidden_log[t].T
            d_input_weight += temp @ self.inputs_log[t].T
            d_Hidden_Feed = self.internal_weight @ temp

        for d in [d_input_weight, d_internal_weight, d_output_weight, d_input_bias, d_output_bias]:
            np.clip(d, -1, 1, out=d)

    
        self.internal_weight -= learning_rate * d_internal_weight
        self.input_weight -= learning_rate * d_input_weight
        self.output_weight -= learning_rate * d_output_weight
        self.input_bias -= learning_rate * d_input_bias
        self.output_bias -= learning_rate * d_output_bias
            
