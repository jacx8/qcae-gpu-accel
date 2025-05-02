import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

from constant import *
from circuit import QuantumLayer


class TorchCircuit(Function):
    @staticmethod
    def forward(ctx, i):
        if not hasattr(ctx, 'QiskitCirc'):
            ctx.QiskitCirc = QuantumLayer(NUM_QUBITS, SIMULATOR, shots=NUM_SHOTS)
 
        device = i.device
        exp_value = ctx.QiskitCirc.run(i.cuda())
        result = torch.tensor([exp_value], device=device)
        ctx.save_for_backward(result, i)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):


        device = grad_output.device
        forward_tensor, i = ctx.saved_tensors
        input_numbers = i
        gradients = torch.zeros(0, device=device)
        
        for k in range(NUM_QUBITS):
            shift_right = input_numbers.detach().clone().to(device)
            shift_right[k] = shift_right[k] + SHIFT
            shift_left = input_numbers.detach().clone().to(device)
            shift_left[k] = shift_left[k] - SHIFT
            
            #expectation_right = ctx.QiskitCirc.run(shift_right)
            #expectation_left  = ctx.QiskitCirc.run(shift_left)
            e_r = torch.tensor([ctx.QiskitCirc.run(shift_right)], device=device)
            e_l = torch.tensor([ctx.QiskitCirc.run(shift_left)],  device=device)

            #gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])*2
            gradient = (e_r - e_l * 2.0)
            gradients = torch.cat((gradients, gradient.float()))
            
        result = torch.tensor(gradients, device=device)
        return (result.float() * grad_output.float()).T
