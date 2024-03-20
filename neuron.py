import random
import numpy as np

class Neuron():
    def __init__(self, inputs):
        self.inputs = inputs
        self.weights = [random.uniform(-1, 1) for i in inputs]
        self.bias = random.random()
        self.output = self.calculate_output()

    def sigmoid(self, x):
        # Clip values to avoid overflow in exp
        clipped_x = np.clip(x, -709, 709)  # 709 is chosen based on the float64 limit
        return 1 / (1 + np.exp(-clipped_x))    
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def calculate_output(self):
        weight_calculation = 0

        # Multiply the inputs by the weights
        for num, _ in enumerate(self.inputs):
            weight_calculation += self.inputs[num] * self.weights[num]
        
        self.z = weight_calculation + self.bias

        # Add the bias
        output = self.sigmoid(self.z)

        return output
    
    def derived_sigma_calculation(self):
        weight_calculation = 0

        # Multiply the inputs by the weights
        for num, _ in enumerate(self.inputs):
            weight_calculation += self.inputs[num] * self.weights[num]
        
        # Add the bias
        output = self.sigmoid_derivative(weight_calculation + self.bias)

        return output