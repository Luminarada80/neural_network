import random
from neuron import Neuron
from layer import StartingLayer, HiddenLayer, FinalLayer
from keras.datasets import mnist
from matplotlib import pyplot
import numpy as np

def cost_function(predicted_array, actual_array):
    cost = 0
    for i, _ in enumerate(predicted_array):
        cost += (predicted_array[i] - actual_array[i]) ** 2
    
    return cost

# def backpropagation():
#     # what number do we want?


# downloading mnist dataset
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# # 60k images in training set, 10k images in testing set
# print('X_train: ' + str(train_x.shape))
# print('Y_train: ' + str(train_y.shape))
# print('X_test:  '  + str(test_x.shape))
# print('Y_test:  '  + str(test_y.shape))

# # Plotting the dataset to confirm
# for i in range(9):  
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(train_x[i], cmap=pyplot.get_cmap('gray'))
# pyplot.show()

total_cost = []
cost_functions = []
for image_num, image in enumerate(train_x[0:1]):
    starting_inputs = np.array(train_x[image_num]).flatten()
    # print(starting_inputs)

    layer_1 = StartingLayer(10, starting_inputs)

    # print('Layer 1:')
    # for node_num, node_output in enumerate(layer_1.calculate_outputs()):
    #     print(f'\tNode {node_num} output: {node_output}')

    layer_2 = HiddenLayer(10, layer_1)

    # print('Layer 2:')
    # for node_num, node_output in enumerate(layer_2.calculate_outputs()):
    #     print(f'\tNode {node_num} output: {node_output}')

    layer_3 = HiddenLayer(10, layer_2)

    print('Layer 3:')
    for node_num, node_output in enumerate(layer_3.calculate_outputs()):
        print(f'\tNode {node_num} output: {node_output}')

    final_layer = FinalLayer(10, layer_3)

    predicted_array = final_layer.calculate_outputs()

    predicted_value = np.argmax(predicted_array)

    actual_value = train_y[image_num]

    actual_value_array = [1 if i == train_y[0] else 0 for i in range(len(predicted_array))]

    total_cost.append(cost_function(predicted_array, actual_value_array))

    # Figures out how much each value in the array should increase or decrease
    difference_array = []
    for i, _ in enumerate(predicted_array):
        difference_array.append((actual_value_array[i] - predicted_array[i]) ** 2)

    print(f'\nActivation array: \n{predicted_array}\n')
    print(f'Actual value array: \n{actual_value_array}\n')
    print(f'Difference_array: \n{difference_array}\n')

    # We want to modify the weights previous layer proportional to the difference array
    # For each neuron in layer 3, add the difference between the desired difference and 
    layer_weight_changes = []
    for neuron in layer_3.neurons:
        weight_change = 0
        for digit_num, digit in enumerate(difference_array):
            weight_change = neuron.weights[digit_num] + digit
        layer_weight_changes.append(weight_change)

    total_cost = np.sum(total_cost)

    # Append the weights for each neuron
    weight_matrix = [] # w
    activation_array = [] # a(L-1)
    for neuron in layer_3.neurons:
        weight_matrix.append([weight for weight in neuron.weights])
        activation_array.append(neuron.calculate_output())

    # print(f'Predicted vs actual: {predicted_value}, {train_y[0]}')
    learning_rate = 0.01  # Example learning rate

    # print(f'Cost = {cost_function(predicted_array, actual_value_array)}')
    layers = [final_layer, layer_3, layer_2, layer_1]

    # Compute the backpropagation for each layer
    for layer_num, layer in enumerate(layers):

        # Finding the mathematical variables
        a_L = np.array(layer.calculate_outputs()) # Activation of the layer
        a_L_1 = np.array(layer.inputs) # Activation of the previous layer
        y = np.array(actual_value_array) # Actual values for the final 
        z_L = np.array(layer.z_L) # (weights * inputs) + bias
        C_0 = np.sum(np.array(difference_array)) # Cost array (actual value - predicted value for each output)
        
        # Derivatives for finding the how the weights of the previous layer affect cost
        # dC_0 / da_L
        dC_0_da_L = 2 * (a_L - y)

        # da_L / dz_L
        da_L_dz_L = [neuron.derived_sigma_calculation() for neuron in layer.neurons]

        # dz_L / dw_L
        dz_L_dw_L = a_L_1

        # da_L / da_L_1
        

        print('dC0 / da_L')
        print(dC_0_da_L)

        print('\nda_L / dz_L')
        print(da_L_dz_L)

        print('\ndz_L / da_L_1')
        print(dz_L_da_L_1)
        # Derivative for finding how the bias affects cost
        # bias_cost = a_L_1 * da_L_dz_L * dC_0_da_L

        gradient_wrt_weights = np.dot(dC_0_da_L, da_L_dz_L) * dz_L_dw_L

        cost_functions.append(affect_of_weight)

# Average all the costs for each training example
np.average(cost_functions)

print(f'\nTOTAL COST')
print(f'\t{total_cost}')

print(f'\nAVERAGE COST')
print(f'\t{np.average(total_cost)}')






