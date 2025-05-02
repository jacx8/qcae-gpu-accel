NUM_QUBITS = 2
NUM_CIRCUITS = 6
SIMULATOR = 'qasm_simulator'
NUM_SHOTS = 1000
SHIFT = 0.9
LR = 0.001
IMAGE_SIZE = 32

import torch

    
def to_numbers(tensor_list):
    num_list = []
    for tensor in tensor_list:
        num_list += [tensor.item()]
    return num_list
