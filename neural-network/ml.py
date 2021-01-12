import numpy as np

class Activation:
    def ReLU(self):
        self.output = np.maximum(0, self.output)
    
    def Softmax(self):
        exp_values = np.exp(self.output - np.max(self.output, axis=1, keepdims=True)) 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class LayerDense (Activation):
    def __init__(self, n_inputs, n_neurons, activation):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = getattr(self, activation)
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.activation()

if __name__ == '__main__':
    import spiral

    X, y = spiral.create_data(100, 3)

    a = LayerDense(2,3,'ReLU')
    a.forward(X)

    b = LayerDense(3,8,'ReLU')
    b.forward(a.output)

    c = LayerDense(8,3,'Softmax')
    c.forward(b.output)

    print(c.output[:5])


    
    