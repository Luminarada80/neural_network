from neuron import Neuron
import numpy as np

class StartingLayer():
    def __init__(self, num_neurons, inputs):
        self.inputs = inputs
        self.neurons = [Neuron(self.inputs) for neuron in range(num_neurons)]
        self.z_L = [neuron.z for neuron in self.neurons]
        
    def calculate_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output())
        
        return outputs

class HiddenLayer():
    def __init__(self, num_neurons, previous_layer):
        self.inputs = previous_layer.calculate_outputs()
        self.neurons = [Neuron(self.inputs) for neuron in range(num_neurons)]
        self.z_L = [neuron.z for neuron in self.neurons]
    
    def calculate_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output())
        
        return outputs

class FinalLayer():
    def __init__(self, num_outputs, previous_layer):
        self.inputs = previous_layer.calculate_outputs()
        self.neurons = [Neuron(self.inputs) for neuron in range(num_outputs)]
        self.z_L = [neuron.z for neuron in self.neurons]

    def calculate_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output())

        return outputs